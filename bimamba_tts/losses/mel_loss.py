import torch
import torch.nn as nn


class MelLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mae = nn.L1Loss()

    def forward(
        self,
        mel_pred: torch.Tensor,
        mel_target: torch.Tensor,
        mel_postnet: torch.Tensor = None,
        mask: torch.Tensor = None,
    ) -> dict:
        if mask is not None:
            mask = mask.unsqueeze(-1).expand_as(mel_pred)
            mel_pred = mel_pred * mask
            mel_target = mel_target * mask
            if mel_postnet is not None:
                mel_postnet = mel_postnet * mask

        mel_loss = self.mae(mel_pred, mel_target)

        losses = {"mel": mel_loss}

        if mel_postnet is not None:
            postnet_loss = self.mae(mel_postnet, mel_target)
            losses["postnet"] = postnet_loss

        return losses
