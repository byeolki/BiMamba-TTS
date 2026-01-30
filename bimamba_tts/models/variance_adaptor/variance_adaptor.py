import torch
import torch.nn as nn

from .duration_predictor import DurationPredictor
from .energy_predictor import EnergyPredictor
from .length_regulator import LengthRegulator
from .pitch_predictor import PitchPredictor


class VarianceAdaptor(nn.Module):
    def __init__(
        self,
        d_model: int,
        duration_predictor_params: dict,
        pitch_predictor_params: dict,
        energy_predictor_params: dict,
    ):
        super().__init__()

        self.duration_predictor = DurationPredictor(
            d_model, **duration_predictor_params
        )
        self.pitch_predictor = PitchPredictor(d_model, **pitch_predictor_params)
        self.energy_predictor = EnergyPredictor(d_model, **energy_predictor_params)
        self.length_regulator = LengthRegulator()

    def forward(
        self,
        x: torch.Tensor,
        duration_target: torch.Tensor = None,
        pitch_target: torch.Tensor = None,
        energy_target: torch.Tensor = None,
        max_len: int = None,
        duration_control: float = 1.0,
        pitch_control: float = 1.0,
        energy_control: float = 1.0,
    ):
        duration_pred = self.duration_predictor(x)

        if duration_target is not None:
            duration = duration_target
        else:
            duration = torch.exp(duration_pred) - 1
            duration = duration * duration_control
            duration = torch.clamp(torch.round(duration), min=0).long()

        x = self.length_regulator(x, duration, max_len)

        pitch_pred = self.pitch_predictor(x)

        if pitch_target is not None:
            pitch_emb = self.pitch_predictor.get_pitch_embedding(pitch_target)
        else:
            pitch_adjusted = pitch_pred * pitch_control
            pitch_emb = self.pitch_predictor.get_pitch_embedding(pitch_adjusted)

        energy_pred = self.energy_predictor(x)

        if energy_target is not None:
            energy_emb = self.energy_predictor.get_energy_embedding(energy_target)
        else:
            energy_adjusted = energy_pred * energy_control
            energy_emb = self.energy_predictor.get_energy_embedding(energy_adjusted)

        x = x + pitch_emb + energy_emb

        return x, duration_pred, pitch_pred, energy_pred
