from torch import nn
from torch import Tensor

from patchTST.layer.TSTEncoderLayer import TSTEncoderLayer

class TSTEncoder(nn.Module):
    def __init__(self,
                 d_model,
                 n_heads,
                 n_layers=1):
        super(TSTEncoder, self).__init__()
        
        layer_list = []
        for _ in range(n_layers):
            layer = TSTEncoderLayer(d_model, n_heads)
            layer_list.append(layer)
        
        self.layers = nn.ModuleList(layer_list)
    
    def forward(self, X: Tensor):
        for layer in self.layers:
            X = layer(X)
        
        return X