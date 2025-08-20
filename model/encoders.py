import torch
import torch.nn as nn
import torch.nn.functional as F
from rotary_embedding_torch import RotaryEmbedding

class MultiheadAttentionWithROPE(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.3, batch_first=None):
        super(MultiheadAttentionWithROPE, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

        # Ensure the embedding dimension can be split into the number of heads
        assert embed_dim % num_heads == 0
        self.head_dim = embed_dim // num_heads

        # Define linear layers for query, key, value projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.bias_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Instantiate rotary embedding
        self.rotary_emb = RotaryEmbedding(dim=self.head_dim)

        self.beta = nn.Parameter(torch.zeros(self.num_heads))

    def forward(self, query, key, value, bias = None, key_padding_mask = None):
        batch_size, seq_len, embed_dim = query.size()

        # Linear projections for query, key, value
        q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply rotary embeddings to queries and keys
        q = self.rotary_emb.rotate_queries_or_keys(q)  # Apply ROPE to queries
        k = self.rotary_emb.rotate_queries_or_keys(k)  # Apply ROPE to keys
        # Scaled dot-product attention
        # q:(batch_size, num_heads, seq_len, head_dim)
        # k.transpose(-2,-1):(batch_size, num_heads, head_dim, seq_len)
        attn_weights = torch.matmul(q, k.transpose(-2, -1))  #(bacth_size,num_heads,seq_len,seq_len)

        if bias is not None:
          b = self.bias_proj(bias).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
          bias_matrix = torch.matmul(b, k.transpose(-2, -1))
          beta = self.beta.view(1, self.num_heads, 1, 1)
          attn_weights = attn_weights + beta * bias_matrix

        attn_weights = attn_weights / (self.head_dim ** 0.5)

        # Optional key_padding_mask (e.g., for padding or causal masking)
        if key_padding_mask is not None:
            # key_padding_mask shape: (batch_size, seq_len) -> need to expand for multi-head attention
            # We expand it to (batch_size, 1, 1, seq_len) and apply it across all heads and queries
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # Shape: (batch_size, 1, 1, seq_len)
            attn_weights = attn_weights.masked_fill(key_padding_mask == True, float('-inf'))

        # Softmax over the attention weights
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Compute attention output
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project back to the original embedding dimension
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights

class FeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim , activation , dropout=0.3):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.activation = activation
        self.dropout=nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class BGCEncoder(nn.Module):
    def __init__(self, 
                 esm_size=1280, 
                 gearnet_size=3072, 
                 attention_dim=512, 
                 num_heads=8,
                 activation=nn.GELU(), 
                 dropout=0.3):
        super(BGCEncoder, self).__init__()
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.activation = activation
        self.esm_projection = nn.Linear(esm_size, attention_dim)
        self.gearnet_projection = nn.Linear(gearnet_size, attention_dim)
        self.self_attention = MultiheadAttentionWithROPE(attention_dim, num_heads, dropout=dropout,batch_first=True)
        self.feedforward = FeedForward(attention_dim, attention_dim*2, self.activation, dropout)
        self.norm1 = nn.LayerNorm(attention_dim)
        self.norm2 = nn.LayerNorm(attention_dim)

    def forward(self, pros, mask, structure = None):
        '''
        pros: (batch_size, sequence_length, input_dim)
        sequence_length: The enzyme count from the BGC with the highest number in this batch.
        class_indices: (batch_size, num_classes=6)
        structure:(batch_size,sequence_length,3072)
        mask: (batch,sequence_length). the mask of padded enzymes in a BGC
        '''

        # (batch_size, sequence_length, esm_size = 1280)->(batch_size, sequence_length, attention_dim)
        pros = self.activation(self.esm_projection(pros)) * ~mask[..., None] 
        if structure is not None:
          structure = self.activation(self.gearnet_projection(structure)) # (batch_size, sequence_length, gearnet_size = 3072)->(batch_size, sequence_length, attention_dim)
        
        self_attn_output, _ = self.self_attention(pros, pros, pros, structure, key_padding_mask = mask) #mask:（batch,seqlen）
        self_attn_output * ~mask[..., None]
        self_attn_output = self.norm1(self_attn_output + pros) #self_attn_out: (batch_size, sequence_length, attention_dim)
        residue = self_attn_output
        self_attn_output = self.feedforward(self_attn_output) 
        self_attn_output = self.norm2(self_attn_output+residue) 

        return self_attn_output * ~mask[..., None]

class SmilesEncoder(nn.Module):
    def __init__(self, 
                 attention_dim=512, 
                 num_heads=8, 
                 vocab_size=138,
                 activation = nn.GELU(),
                 dropout=0.3):
        super(SmilesEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size+1, attention_dim, padding_idx = vocab_size) 
        self.activation=activation
        self.feedforward = FeedForward(attention_dim, attention_dim*2, self.activation, dropout)
        self.norm1 = nn.LayerNorm(attention_dim)
        self.norm2 = nn.LayerNorm(attention_dim)

        self.alpha = torch.zeros(1, device=next(self.parameters()).device)
        self.self_attention = MultiheadAttentionWithROPE(attention_dim, num_heads, dropout=dropout, batch_first=True) 

    def forward(self, subs, mask=None):
        # Self-Attention on pros
        subs_embedded=self.embedding(subs)

        # Self-Attention on subs
        self_attn_output, _ = self.self_attention(subs_embedded, subs_embedded, subs_embedded, key_padding_mask=mask)
        self_attn_output * ~mask[..., None]
        self_attn_output = self.norm1(self_attn_output + subs_embedded)
        residue = self_attn_output
        self_attn_output = self.feedforward(self_attn_output)
        self_attn_output = self.norm2(self_attn_output+residue)

        return self_attn_output * ~mask[..., None]