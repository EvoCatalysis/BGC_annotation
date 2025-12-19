import torch.nn as nn
import torch

from model.encoders import FeedForward, BGCEncoder, SmilesEncoder

class BGCClassifier(nn.Module):
    def __init__(self, 
                 esm_size = 1280, 
                 gearnet_size = 3072, 
                 num_classes = 6, 
                 attention_dim = 512, 
                 num_heads = 8, 
                 dropout = 0.3):
        super().__init__()
        self.num_classes = num_classes

        self.activation = nn.GELU()
        self.BGCEncoder = BGCEncoder(esm_size, gearnet_size, attention_dim, num_heads, self.activation, dropout)
        self.class_embedding = nn.Embedding(num_classes, attention_dim)
        self.cross_attention = nn.MultiheadAttention(embed_dim = attention_dim, num_heads = num_heads, dropout = dropout, batch_first = True)
        self.feedforward = FeedForward(attention_dim, attention_dim * 2, self.activation, dropout)
        self.norm1 = nn.LayerNorm(attention_dim)
        self.norm2 = nn.LayerNorm(attention_dim)
        #self.fc = nn.Linear(attention_dim, 1)
        self.fcs = nn.ModuleList([
            nn.Linear(attention_dim, 1) for _ in range(num_classes)
        ])
        
    def forward(self, pros, class_indices, mask, structure = None):
        '''
        Args:
            pros: (batch_size, sequence_length, input_dim)
            sequence_length: The enzyme count from the BGC with the highest number in this batch.
            class_indices: (batch_size, num_classes=6)
            structure:(batch_size,sequence_length,3072)
            mask: (batch,sequence_length). the mask of padded enzymes in a BGC
        '''
        # (batch_size, num_classes) => (batch_size, num_classes, attention_dim)
        class_embeddings = self.class_embedding(class_indices)
        # (batch_size, len_BGC, attention_dim)
        self_attn_output = self.BGCEncoder(pros, mask, structure)
        
        # cross attention: class_embedding (Query) to BGC embedding (Key and value)
        # cross_attn_weights: (batch_size, num_heads, q: num_classes, k: sequence_length)
        # cross_attn_output: (batch_size, num_classes, attention_dim)
        cross_attn_output, cross_attn_weights = self.cross_attention(class_embeddings, 
                                                                     self_attn_output, 
                                                                     self_attn_output,
                                                                     key_padding_mask = mask)
        cross_attn_output = self.norm1(cross_attn_output + class_embeddings)
        ff_out = self.feedforward(cross_attn_output)
        ff_out = self.norm2(cross_attn_output + ff_out) #(batch_size, num_classes, attention_dim)
        
        # (batch_size, num_classes)
        #logits = self.fc(ff_out).squeeze(-1) 
        logits_list = []
        for i in range(self.num_classes):
            class_embedding = ff_out[:, i, :]  # (batch_size, attention_dim)
            logit = self.fcs[i](class_embedding) # (batch_size, 1)
            logits_list.append(logit)

        logits = torch.stack(logits_list, dim=1).squeeze(-1) # (batch_size, num_classes, 1)

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
        # mask1: (batch_size, len_BGC)
        # mask2: (batch_size, len_smiles)
        # Self-Attention on pros
        # (batch_size, len_BGC, attention_dim)
        self_attn_output1 = self.BGCEncoder(pros, mask1, structure)

        # Self-Attention on subs
        # (batch_size, len_smiles, attention_dim)
        self_attn_output2 = self.SmilesEncoder(subs, mask2)

        # Cross-Attention: pros (Query) to subs (Key/Value)
        # (batch_size, len_BGC, attention_dim)
        cross_attn_output, cross_attn_matrix = self.cross_attention(self_attn_output1, 
                                                                    self_attn_output2, 
                                                                    self_attn_output2, 
                                                                    key_padding_mask = mask2, 
                                                                    average_attn_weights = average_cross_attn)
        
        cross_attn_output *= ~mask1[..., None]
        cross_attn_output = self.norm1(cross_attn_output + self_attn_output1)
        ff_out = self.feedforward(cross_attn_output)
        ff_out = self.norm2(cross_attn_output + ff_out)
        ff_out *= ~mask1[..., None]
        # Global average pooling
        
        # (batch_size, seq_len, attention_dim) => (batch_szie, attention_dim)
        out = ff_out.sum(dim = 1) / ((~mask1).sum(dim = -1)[..., None] + 1e-9) 
        # Concatenate and classify
        out = self.activation(self.projection(out))
        out = self.dropout(out)
        
        # logits: [batch_size, 1]
        out = self.fc(out)  
        return out, cross_attn_matrix