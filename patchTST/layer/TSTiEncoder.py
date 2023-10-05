import torch
from torch import nn
from torch import Tensor

from patchTST.layer.TSTEncoder import TSTEncoder

class TSTiEncoder(nn.Module):
    def __init__(self,
                 device,
                 patch_num,
                 patch_len,
                 dropout=0.,
                 d_model=128,
                 n_heads=16):
        super(TSTiEncoder, self).__init__()
        self.patch_num = patch_num
        self.patch_len = patch_len
        
        self.W_P = nn.Linear(patch_len, d_model)
        
        # positional_encoding("zeros")
        W_pos = torch.empty((patch_num, d_model))
        nn.init.uniform_(W_pos, -0.02, 0.02).to(device)
        self.W_pos = nn.Parameter(W_pos, requires_grad=True)
        
        self.dropout = nn.Dropout(dropout)
        self.encoder = TSTEncoder(d_model, n_heads)
        
    def forward(self, X: Tensor):
        nvars = X.shape[1]
        
        X = X.permute(0, 1, 3, 2)
        X = self.W_P(X)
        
        # u: [batch_size * nvars, patch_num, d_model]
        u = torch.reshape(X, (X.shape[0] * X.shape[1], X.shape[2], X.shape[3]))
        u = self.dropout(u + self.W_pos)
        
        # encoder
        Z = self.encoder(u)
        Z = torch.reshape(Z, (-1, nvars, Z.shape[-2], Z.shape[-1]))
        
        # Z: [batch_size, nvars, d_model, patch_num]
        Z = Z.permute(0, 1, 3, 2)
        
        return Z
