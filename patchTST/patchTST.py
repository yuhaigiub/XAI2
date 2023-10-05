from torch import nn
from torch import Tensor

from patchTST.layer.PatchTST_Backbone import PatchTST_Backbone

class PatchTST(nn.Module):
    def __init__(self, device, context_window, target_window, patch_len, stride):
        '''
            context_window: sequence length
            target_window: predict length
        '''
        super(PatchTST, self).__init__()
        self.model = PatchTST_Backbone(device, context_window, target_window, patch_len, stride)
        
    def forward(self, X: Tensor): # x: [batch_size, input_length, channels]
        X = X.permute(0, 2, 1) # x: [batch_size, channels, input_length]
        X = self.model(X)
        X = X.permute(0, 2, 1) # x: [batch_size, input_length, channels]
        return X