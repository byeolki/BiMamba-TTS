import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.append(str(Path(__file__).parent.parent.parent))

from bimamba_tts.models.bimamba_tts import BiMambaTTS
from bimamba_tts.utils import AudioProcessor, Phonemizer, load_checkpoint
from bimamba_tts.vocoder.hifigan import HiFiGAN


def synthesize(
    text,
    model,
    phonemizer,
    vocoder,
    device,
    duration_control=1.0,
    pitch_control=1.0,
    energy_control=1.0,
):
    model.eval()

    phone_sequence = phonemizer.text_to_sequence(text)
    phone_tensor = torch.LongTensor(phone_sequence).unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = model(
            text=phone_tensor,
            duration_control=duration_control,
            pitch_control=pitch_control,
            energy_control=energy_control,
        )

    mel = predictions["mel_postnet"][0].cpu().numpy()

    if vocoder is not None:
        wav = vocoder.inference(mel)
        return mel, wav
    else:
        return mel, None


def main():
    parser = argparse.ArgumentParser(description="BiMamba-TTS Inference")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--output", type=str, default="output.wav")
    parser.add_argument("--duration_control", type=float, default=1.0)
    parser.add_argument("--pitch_control", type=float, default=1.0)
    parser.add_argument("--energy_control", type=float, default=1.0)
    parser.add_argument("--vocoder_checkpoint", type=str, default=None)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    config = OmegaConf.to_container(config, resolve=True)

    device = torch.device(config["device"])

    model = BiMambaTTS(config).to(device)
    checkpoint = load_checkpoint(args.checkpoint, device=device)
    model.load_state_dict(checkpoint["model"])

    phonemizer = Phonemizer()
    audio_processor = AudioProcessor(config["audio"])

    vocoder = None
    if args.vocoder_checkpoint:
        vocoder = HiFiGAN(args.vocoder_checkpoint, device)

    mel, wav = synthesize(
        args.text,
        model,
        phonemizer,
        vocoder,
        device,
        args.duration_control,
        args.pitch_control,
        args.energy_control,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.save(str(output_path).replace(".wav", "_mel.npy"), mel)

    if wav is not None:
        audio_processor.save_wav(str(output_path), wav)
        print(f"Synthesized audio saved to {output_path}")
    else:
        print(f"Mel spectrogram saved to {output_path.replace('.wav', '_mel.npy')}")
        print("Note: Vocoder not provided. Only mel spectrogram is saved.")


if __name__ == "__main__":
    main()
