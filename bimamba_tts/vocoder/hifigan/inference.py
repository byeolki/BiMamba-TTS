import numpy as np
import torch

from .models import HiFiGAN


def load_hifigan(checkpoint_path=None, device="cuda"):
    return HiFiGAN(checkpoint_path, device)


def mel_to_audio(mel, vocoder):
    return vocoder.inference(mel)
