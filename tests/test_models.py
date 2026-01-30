import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parent.parent))

from bimamba_tts.models import BiMambaTTS
from bimamba_tts.models.common import BiMambaBlock, MambaBlock


def test_mamba_block():
    batch_size = 2
    seq_len = 10
    d_model = 256

    block = MambaBlock(d_model=d_model, d_state=16, d_conv=4, expand=2)
    x = torch.randn(batch_size, seq_len, d_model)
    output = block(x)

    assert output.shape == x.shape


def test_bimamba_block():
    batch_size = 2
    seq_len = 10
    d_model = 256

    block = BiMambaBlock(d_model=d_model, d_state=16, d_conv=4, expand=2)
    x = torch.randn(batch_size, seq_len, d_model)
    output = block(x)

    assert output.shape == x.shape


def test_bimamba_tts():
    config = {
        "model": {
            "vocab_size": 158,
            "max_seq_len": 1000,
            "encoder": {
                "n_layers": 2,
                "d_model": 256,
                "d_state": 16,
                "d_conv": 4,
                "expand": 2,
                "dropout": 0.1,
            },
            "decoder": {
                "n_layers": 2,
                "d_model": 256,
                "d_state": 16,
                "d_conv": 4,
                "expand": 2,
                "dropout": 0.1,
            },
            "variance_adaptor": {
                "duration_predictor": {
                    "n_layers": 2,
                    "kernel_size": 3,
                    "dropout": 0.3,
                },
                "pitch_predictor": {
                    "n_layers": 2,
                    "kernel_size": 3,
                    "dropout": 0.3,
                    "n_bins": 256,
                    "pitch_min": 4.0,
                    "pitch_max": 7.0,
                },
                "energy_predictor": {
                    "n_layers": 2,
                    "kernel_size": 3,
                    "dropout": 0.3,
                    "n_bins": 256,
                    "energy_min": -4.5,
                    "energy_max": 6.0,
                },
            },
        },
        "audio": {
            "n_mel_channels": 80,
        },
    }

    model = BiMambaTTS(config)

    batch_size = 2
    text_len = 20
    mel_len = 100

    text = torch.randint(0, 158, (batch_size, text_len))
    durations = torch.randint(1, 10, (batch_size, text_len))
    pitches = torch.randn(batch_size, mel_len)
    energies = torch.randn(batch_size, mel_len)

    outputs = model(
        text=text,
        duration_target=durations,
        pitch_target=pitches,
        energy_target=energies,
        max_len=mel_len,
    )

    assert "mel_out" in outputs
    assert "mel_postnet" in outputs
    assert "duration_pred" in outputs
    assert "pitch_pred" in outputs
    assert "energy_pred" in outputs


if __name__ == "__main__":
    test_mamba_block()
    test_bimamba_block()
    test_bimamba_tts()
    print("All tests passed!")
