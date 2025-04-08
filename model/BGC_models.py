import torch
import torch.nn as nn
import torch.nn.functional as F

from model.encoders import FeedForward, MultiheadAttentionWithROPE, BGCEncoder, SmilesEncoder

class BGCClassifier(nn.Module):
    def __init__(self, 
                 esm_size=1280, 
                 gearnet_size=3072, 
                 num_classes=6, 
                 attention_dim=512, 
                 num_heads=8, 
                 dropout=0.3):
        super().__init__()

        self.activation = nn.GELU()
        self.BGCEncoder = BGCEncoder(esm_size, gearnet_size, attention_dim, num_heads, self.activation, dropout)
        self.class_embedding = nn.Embedding(num_classes, attention_dim)
        self.cross_attention = nn.MultiheadAttention(embed_dim = attention_dim, num_heads = num_heads, dropout = dropout, batch_first = True)
        self.feedforward = FeedForward(attention_dim, attention_dim * 2, self.activation, dropout)
        self.norm1 = nn.LayerNorm(attention_dim)
        self.norm2 = nn.LayerNorm(attention_dim)
        self.fc = nn.Linear(attention_dim, 1)

    def forward(self, pros, class_indices, mask, structure = None):
        '''
        pros: (batch_size, sequence_length, input_dim)
        sequence_length: The enzyme count from the BGC with the highest number in this batch.
        class_indices: (batch_size, num_classes=6)
        structure:(batch_size,sequence_length,3072)
        mask: (batch,sequence_length). the mask of padded enzymes in a BGC
        '''

        class_embeddings = self.class_embedding(class_indices) #(batch_size, num_classes)->(batch_size, num_classes, attention_dim)
        self_attn_output = self.BGCEncoder(pros, mask, structure)
        
        #cross_attn_weights: (batch_size, num_heads, q: num_classes, k: sequence_length)
        #cross_attn_output: (batch_size,num_classes, attention_dim)
        cross_attn_output, cross_attn_weights = self.cross_attention(class_embeddings, self_attn_output, self_attn_output)
        cross_attn_output = self.norm1(cross_attn_output + class_embeddings)
        ff_out = self.feedforward(cross_attn_output)
        ff_out = self.norm2(cross_attn_output+cross_attn_output)
        
        # (batch_size, num_classes)
        logits = self.fc(ff_out).squeeze(-1) 

        return logits, cross_attn_weights

class ProductMatching(nn.Module):
    def __init__(self, 
                 esm_size = 1280, 
                 gearnet_size = 3072, 
                 attention_dim = 512, 
                 num_heads = 8, 
                 vocab_size = 138, 
                 dropout = 0.3):
        super().__init__()

        self.projection = nn.Linear(attention_dim, attention_dim)
        self.activation = nn.GELU()

        self.BGCEncoder = BGCEncoder(esm_size, gearnet_size, attention_dim, num_heads, self.activation, dropout)
        self.SmilesEncoder = SmilesEncoder(attention_dim, num_heads, vocab_size, self.activation, dropout)
        self.feedforward = FeedForward(attention_dim, attention_dim*2, self.activation, dropout)
        self.norm1 = nn.LayerNorm(attention_dim)
        self.norm2 = nn.LayerNorm(attention_dim)

        self.cross_attention = nn.MultiheadAttention(attention_dim, num_heads, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(attention_dim, 1)  
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()  

    def forward(self, pros, subs, structure = None, mask1=None, mask2=None, average_cross_attn=True):
        # Self-Attention on pros
        self_attn_output1 = self.BGCEncoder(pros, mask1, structure) #pros:[batch,seqlen,embed_size] #mask:[batch,seqlen]
        # Self-Attention on subs
        self_attn_output2 = self.SmilesEncoder(subs, mask2) #subs:[batch,seqlen,embed_size] #mask:[batch,seqlen]
        # Cross-Attention: pros (Query) to subs (Key/Value)
        cross_attn_output, cross_attn_matrix = self.cross_attention(self_attn_output1, self_attn_output2, self_attn_output2, key_padding_mask=mask2, average_attn_weights = average_cross_attn)
        cross_attn_output = self.norm1(cross_attn_output + self_attn_output1)
        ff_out = self.feedforward(cross_attn_output)
        ff_out = self.norm2(cross_attn_output + ff_out)
        # Global average pooling
        out = torch.mean(ff_out, dim=1)  # [batch_size, seq_len, attention_dim]->[batch_szie, attention_dim]
        # Concatenate and classify
        out = self.activation(self.projection(out))
        out = self.dropout(out)
        out = self.fc(out)  # [batch_size, 1]

        return out, cross_attn_matrix