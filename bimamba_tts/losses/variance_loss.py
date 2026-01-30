import torch
import torch.nn as nn


class VarianceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None
    ) -> torch.Tensor:
        if mask is not None:
            pred = pred * mask
            target = target * mask

        loss = self.mse(pred, target)

        return loss
