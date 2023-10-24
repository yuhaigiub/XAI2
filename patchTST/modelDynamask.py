import torch
from torch import nn
from patchTST.patchTST import PatchTST
from pertubate import FadeMovingAverage
import util

class Model():
    def __init__(self, device, context_window, target_window, patch_len, stride):
        self.blackbox = PatchTST(device, context_window, target_window, patch_len, stride).to(device)
        self.pertubation = FadeMovingAverage(device)
        self.loss = util.masked_mae
        
        initial_mask_coeff = 0.5
        learning_rate = 1.0e-1
        momentum = 0.9
        self.mask_tensor = initial_mask_coeff * torch.ones(size=X.shape, device=self.device)
        # Create a copy of the mask that is going to be trained, the optimizer and the history
        mask_tensor_new = self.mask_tensor.clone().detach().requires_grad_(True)
        
        self.optimizer = torch.optim.SGD([mask_tensor_new], lr=learning_rate, momentum=momentum)
        # self.clip = 5
        self.clip = None
    
    def f(self, X: torch.Tensor):
        X = X.squeeze(-1)
        X = self.blackbox(X)
        X = X.unsqueeze(-1)
        return X
    
    def train(self, X, y_real):
        n_epochs = 200
        
        y_pred = self.f(X)
    
        loss = self.loss(y_pred, y_real, 0.0)
        loss.backward()
        
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.patchTST.parameters(), self.clip)
        # self.optimizer.step()
        
        self.saliency.data = self.saliency.data.clamp(0, 1)
        
        mape = util.masked_mape(y_pred, y_real, 0.0).item()
        rmse = util.masked_rmse(y_pred, y_real, 0.0).item()
        
        return loss.item(), mape, rmse

    def eval():
        pass