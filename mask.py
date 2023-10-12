import torch
import torch.optim as optim
from pertubate import Pertubation

import numpy as np

class Mask:
    def __init__(self, 
                 pertubation: Pertubation,
                 device):
        self.device = device
        self.pertubation = pertubation
    
    def fit(self, 
            X: torch.Tensor, 
            f, 
            loss_function,
            target: torch.Tensor | None = None,
            keep_ratio: float = 0.5,
            initial_mask_coeff: float = 0.5,
            size_reg_factor_init: float = 0.5,
            size_reg_factor_dilation: float = 100.0,
            time_reg_factor: float = 0.0,
            learning_rate: float = 1.0e-1,
            momentum: float = 0.9):
        reg_factor = size_reg_factor_init
        reg_multiplicator = np.exp(np.log(size_reg_factor_dilation) / n_epoch)
        
        self.f = f
        self.X = X
        self.n_epoch = n_epoch
        
        self.loss_fn = loss_function
        
        self.mask_tensor = initial_mask_coeff * torch.ones(size=X.shape, device=self.device)
        mask_tensor_new = self.mask_tensor.clone().detach().requires_grad_(True)
        optimizer = optim.SGD([mask_tensor_new], lr=learning_rate, momentum=momentum)
        
        X_pert = self.pertubation.apply(X = X, mask_tensor=mask_tensor_new)
        y_pert = f(X_pert)
        
        error = loss_function(Y_pert, self.target)
        