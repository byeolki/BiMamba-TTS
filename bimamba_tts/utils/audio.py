from typing import Optional

import librosa
import numpy as np
import torch
import torchaudio


class AudioProcessor:
    def __init__(self, config: dict):
        self.sampling_rate = config["sampling_rate"]
        self.filter_length = config["filter_length"]
        self.hop_length = config["hop_length"]
        self.win_length = config["win_length"]
        self.n_mel_channels = config["n_mel_channels"]
        self.mel_fmin = config["mel_fmin"]
        self.mel_fmax = config["mel_fmax"]
        self.max_wav_value = config["max_wav_value"]

        from torchaudio.prototype.pipelines import HIFIGAN_VOCODER_V3_LJSPEECH

        bundle = HIFIGAN_VOCODER_V3_LJSPEECH
        self.mel_transform = bundle.get_mel_transform()

        assert self.sampling_rate == bundle.sample_rate
        assert self.filter_length == self.mel_transform.n_fft
        assert self.hop_length == self.mel_transform.hop_size
        assert self.win_length == self.mel_transform.win_length
        assert self.n_mel_channels == self.mel_transform.n_mels
        assert self.mel_fmin == self.mel_transform.f_min
        assert self.mel_fmax == self.mel_transform.f_max

    def load_wav(self, path: str) -> np.ndarray:
        wav, sr = librosa.load(path, sr=self.sampling_rate)
        if sr != self.sampling_rate:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=self.sampling_rate)
        return wav

    def save_wav(self, path: str, wav: np.ndarray):
        wav = wav * self.max_wav_value
        wav = wav.astype(np.int16)
        import soundfile as sf

        sf.write(path, wav, self.sampling_rate)

    def wav_to_mel(self, wav: np.ndarray) -> np.ndarray:
        if isinstance(wav, np.ndarray):
            wav = torch.FloatTensor(wav)

        if len(wav.shape) == 1:
            wav = wav.unsqueeze(0)

        mel_spec = self.mel_transform(wav)

        return mel_spec.squeeze(0).T.numpy()

    def mel_to_wav(self, mel: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Use HiFi-GAN vocoder for mel to wav conversion")

    def extract_pitch(self, wav: np.ndarray, use_log: bool = True) -> np.ndarray:
        import pyworld as pw

        wav = wav.astype(np.float64)
        f0, t = pw.dio(
            wav,
            self.sampling_rate,
            frame_period=self.hop_length / self.sampling_rate * 1000,
        )
        f0 = pw.stonemask(wav, f0, t, self.sampling_rate)

        if use_log:
            nonzero_indices = f0 > 0
            if nonzero_indices.any():
                f0[nonzero_indices] = np.log(f0[nonzero_indices])

        return f0

    def extract_energy(self, wav: np.ndarray, use_log: bool = True) -> np.ndarray:
        stft = librosa.stft(
            wav,
            n_fft=self.filter_length,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window="hann",
            center=True,
        )

        magnitude = np.abs(stft)
        energy = np.linalg.norm(magnitude, axis=0)

        if use_log:
            energy = np.log(energy + 1e-5)

        return energy

    def normalize_features(
        self,
        feature: np.ndarray,
        mean: Optional[float] = None,
        std: Optional[float] = None,
    ) -> tuple:
        if mean is None:
            mean = np.mean(feature[feature != 0])
        if std is None:
            std = np.std(feature[feature != 0])

        normalized = (feature - mean) / (std + 1e-8)
        return normalized, mean, std
