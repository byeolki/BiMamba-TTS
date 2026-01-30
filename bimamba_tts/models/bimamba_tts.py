import torch
import torch.nn as nn

from .decoder import BiMambaDecoder
from .encoder import BiMambaEncoder
from .variance_adaptor import VarianceAdaptor


class BiMambaTTS(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        model_config = config["model"]
        audio_config = config["audio"]

        self.encoder = BiMambaEncoder(
            vocab_size=model_config["vocab_size"],
            d_model=model_config["encoder"]["d_model"],
            n_layers=model_config["encoder"]["n_layers"],
            d_state=model_config["encoder"]["d_state"],
            d_conv=model_config["encoder"]["d_conv"],
            expand=model_config["encoder"]["expand"],
            dropout=model_config["encoder"]["dropout"],
            max_seq_len=model_config["max_seq_len"],
        )

        self.variance_adaptor = VarianceAdaptor(
            d_model=model_config["encoder"]["d_model"],
            duration_predictor_params=model_config["variance_adaptor"][
                "duration_predictor"
            ],
            pitch_predictor_params=model_config["variance_adaptor"]["pitch_predictor"],
            energy_predictor_params=model_config["variance_adaptor"][
                "energy_predictor"
            ],
        )

        self.decoder = BiMambaDecoder(
            d_model=model_config["decoder"]["d_model"],
            n_layers=model_config["decoder"]["n_layers"],
            d_state=model_config["decoder"]["d_state"],
            d_conv=model_config["decoder"]["d_conv"],
            expand=model_config["decoder"]["expand"],
            n_mel_channels=audio_config["n_mel_channels"],
            dropout=model_config["decoder"]["dropout"],
        )

    def forward(
        self,
        text: torch.Tensor,
        src_mask: torch.Tensor = None,
        mel_mask: torch.Tensor = None,
        duration_target: torch.Tensor = None,
        pitch_target: torch.Tensor = None,
        energy_target: torch.Tensor = None,
        max_len: int = None,
        duration_control: float = 1.0,
        pitch_control: float = 1.0,
        energy_control: float = 1.0,
    ):
        encoder_output = self.encoder(text, src_mask)

        variance_output, duration_pred, pitch_pred, energy_pred = self.variance_adaptor(
            encoder_output,
            duration_target=duration_target,
            pitch_target=pitch_target,
            energy_target=energy_target,
            max_len=max_len,
            duration_control=duration_control,
            pitch_control=pitch_control,
            energy_control=energy_control,
        )

        mel_out, mel_postnet = self.decoder(variance_output, mel_mask)

        return {
            "mel_out": mel_out,
            "mel_postnet": mel_postnet,
            "duration_pred": duration_pred,
            "pitch_pred": pitch_pred,
            "energy_pred": energy_pred,
        }
