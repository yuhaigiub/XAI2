import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor

class SDPAttention(nn.Module):
    def __init__(self,
                 d_model, 
                 n_heads,
                 attention_dropout=0.,
                 lsa=False):
        super(SDPAttention, self).__init__()
        
        self.attention_dropout = nn.Dropout(attention_dropout)
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa
        
    
    def forward(self,
                q: Tensor,
                k: Tensor,
                v: Tensor):
        '''
            Input shape:
                q: [batch_size, n_heads, max_q_len, d_k]
                k: [batch_size, n_heads, d_k, seq_len]
                v: [batch_size, n_heads, seq_len, d_v]
        '''
        
        attn_scores = torch.matmul(q, k) * self.scale
        
        # normalize the attention weights
        # attn_weights: [batch_size, n_heads, max_q_len, q_len]
        attention_weights = F.softmax(attn_scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)
        
        # compute the new values given the attention weights
        # output: [bs, n_heads, max_q_len, d_v]
        output = torch.matmul(attention_weights, v)
        
        return output, attention_weights