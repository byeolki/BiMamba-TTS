import torch
import torch.nn as nn


class HiFiGAN:
    def __init__(self, checkpoint_path=None, device="cuda"):
        self.device = device
        from torchaudio.prototype.pipelines import HIFIGAN_VOCODER_V3_LJSPEECH

        bundle = HIFIGAN_VOCODER_V3_LJSPEECH
        self.model = bundle.get_vocoder()
        self.mel_transform = bundle.get_mel_transform()

        if checkpoint_path and checkpoint_path.endswith(".pth"):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if isinstance(checkpoint, dict) and "generator" in checkpoint:
                self.model.load_state_dict(checkpoint["generator"])
            elif isinstance(checkpoint, dict):
                self.model.load_state_dict(checkpoint)

        self.model = self.model.to(device)
        self.mel_transform = self.mel_transform.to(device)
        self.model.eval()
        self.sample_rate = 22050

    def inference(self, mel):
        if not isinstance(mel, torch.Tensor):
            mel = torch.FloatTensor(mel)

        if len(mel.shape) == 2:
            mel = mel.T.unsqueeze(0)

        mel = mel.to(self.device)

        with torch.no_grad():
            wav = self.model(mel)

        if isinstance(wav, tuple):
            wav = wav[0]

        return wav.squeeze().cpu().numpy()
