from .duration_loss import DurationLoss
from .mel_loss import MelLoss
from .variance_loss import VarianceLoss


class BiMambaTTSLoss:
    def __init__(self, config):
        self.mel_loss = MelLoss()
        self.duration_loss = DurationLoss()
        self.pitch_loss = VarianceLoss()
        self.energy_loss = VarianceLoss()

        self.weights = config["train"]["loss_weights"]

    def __call__(self, predictions, targets, masks):
        mel_losses = self.mel_loss(
            predictions["mel_out"],
            targets["mels"],
            predictions["mel_postnet"],
            masks["mel_masks"],
        )

        duration_loss = self.duration_loss(
            predictions["duration_pred"], targets["durations"], masks["src_masks"]
        )

        pitch_loss = self.pitch_loss(
            predictions["pitch_pred"], targets["pitches"], masks["mel_masks"]
        )

        energy_loss = self.energy_loss(
            predictions["energy_pred"], targets["energies"], masks["mel_masks"]
        )

        total_loss = (
            self.weights["mel"] * mel_losses["mel"]
            + self.weights["postnet"] * mel_losses["postnet"]
            + self.weights["duration"] * duration_loss
            + self.weights["pitch"] * pitch_loss
            + self.weights["energy"] * energy_loss
        )

        return {
            "total": total_loss,
            "mel": mel_losses["mel"],
            "postnet": mel_losses["postnet"],
            "duration": duration_loss,
            "pitch": pitch_loss,
            "energy": energy_loss,
        }


__all__ = ["MelLoss", "DurationLoss", "VarianceLoss", "BiMambaTTSLoss"]
