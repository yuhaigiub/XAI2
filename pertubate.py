import torch
from abc import ABC, abstractmethod


class Pertubation(ABC):
    @abstractmethod
    def __init__(self, device, eps=1.0e-7):
        self.mask_tensor = None
        self.device = device
        self.eps = eps
    
    @abstractmethod
    def apply(self, X, mask_tensor):
        if X is None or mask_tensor is None:
            raise NameError("The mask_tensor should be fitted before or while calling the apply() method")


class FadeMovingAverage(Pertubation):
    def __init__(self, device, eps=1.0e-7):
        super().__init__(device, eps)
    
    def apply(self, X, mask_tensor):
        super().apply(X, mask_tensor)
        
        T = X.shape[0]
        moving_average = torch.mean(X, 0).to(self.device)
        moving_average_tilted = moving_average.repeat(T, 1, 1, 1)
        X_pert = mask_tensor * X + (1 - mask_tensor) * moving_average_tilted
        
        return X_pert