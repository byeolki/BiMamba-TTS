import torch
import torch.nn as nn


class DurationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(
        self,
        duration_pred: torch.Tensor,
        duration_target: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        duration_target = torch.log(duration_target.float() + 1)

        if mask is not None:
            duration_pred = duration_pred * mask
            duration_target = duration_target * mask

        loss = self.mse(duration_pred, duration_target)

        return loss
