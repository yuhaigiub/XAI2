from torch import nn
from torch import Tensor

from patchTST.layer.FlattenHead import FlattenHead
from patchTST.layer.TSTiEncoder import TSTiEncoder

class PatchTST_Backbone(nn.Module):
    def __init__(self,
                 device,
                 context_window,
                 target_window,
                 patch_len,
                 stride,
                 d_model=128):
        super(PatchTST_Backbone, self).__init__()

        self.patch_len = patch_len
        self.stride = stride
        patch_num = int((context_window - patch_len) / stride + 1) # (96 - 16) / 8 + 1 = 11 (Example in PatchTST)
        self.head_nf = d_model * patch_num
        
        self.backbone = TSTiEncoder(device, patch_num, patch_len)
        self.head = FlattenHead(self.head_nf, target_window)

    def forward(self, Z: Tensor):  # Z: [batch_size, nvars, seq_len]
        Z = Z.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # Z: [batch_size, nvars, patch_len, patch_num]
        Z = Z.permute(0, 1, 3, 2)
        
        Z = self.backbone(Z)
        Z = self.head(Z)
        
        return Z