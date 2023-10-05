from torch import nn
from torch import Tensor
from typing import Optional

from patchTST.layer.SDPAttention import SDPAttention

class MultiheadAttention(nn.Module):
    def __init__(self,
                 d_model, 
                 n_heads,
                 projection_dropout = 0.,
                 qkv_bias=True):
        super(MultiheadAttention, self).__init__()
        
        d_k = d_model // n_heads
        d_v = d_model // n_heads

        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)
        
        self.sdp_attention = SDPAttention(d_model, n_heads)
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), 
                                    nn.Dropout(projection_dropout))
        
    def forward(self, 
                Q:Tensor,
                K:Optional[Tensor]=None,
                V:Optional[Tensor]=None):
        batch_size = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q
        
        # Linear (+ split in multiple heads)
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).permute(0, 2, 3, 1)     
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)
        
        output, attention_weights = self.sdp_attention(q_s, k_s, v_s)
        
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v)
        output = self.to_out(output)
        
        return output, attention_weights