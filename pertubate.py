import torch


def apply_gaussian_blur(device, X: torch.Tensor, saliency):
    mean = X[..., 0].mean()
    std = X[..., 0].std()
    noise = torch.empty(*X.shape, device=device)
    noise.normal_(mean, std)
    # low saliency score == important
    X_perturbed = X + (saliency * noise)
    # X_perturbed.to(self.device)
    return X_perturbed
