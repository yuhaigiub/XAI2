from torch import nn
from torch import Tensor

from patchTST.layer.MultiheadAttention import MultiheadAttention

class TSTEncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 n_heads,
                 d_ff=256,
                 bias=True,
                 dropout=0.,
                 activation=nn.GELU):
        super(TSTEncoderLayer, self).__init__()
        assert not d_model % n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.self_attention = MultiheadAttention(d_model, n_heads)
        self.dropout_attention = nn.Dropout(dropout)

        self.feed_forward = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                          activation(),
                                          nn.Dropout(dropout),
                                          nn.Linear(d_ff, d_model, bias=bias))
        self.dropout_feed_forward = nn.Dropout(dropout)
        
    def forward(self, X:Tensor):
        X2, attention_weights = self.self_attention(X)
        X = X + self.dropout_attention(X2)
        
        X2 = self.feed_forward(X)
        X = X + self.dropout_feed_forward(X2)
        
        return X