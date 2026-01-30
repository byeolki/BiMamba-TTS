import argparse
import json
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from bimamba_tts.data import get_text_cleaner
from bimamba_tts.utils import AudioProcessor, Phonemizer


def preprocess_ljspeech(config):
    raw_dir = Path(config["paths"]["raw_dir"])
    preprocessed_dir = Path(config["paths"]["preprocessed_dir"])

    text_dir = preprocessed_dir / "text"
    mel_dir = preprocessed_dir / "mel"
    duration_dir = preprocessed_dir / "duration"
    pitch_dir = preprocessed_dir / "pitch"
    energy_dir = preprocessed_dir / "energy"

    for d in [text_dir, mel_dir, duration_dir, pitch_dir, energy_dir]:
        d.mkdir(parents=True, exist_ok=True)

    audio_processor = AudioProcessor(config["audio"])
    phonemizer = Phonemizer()
    text_cleaner = get_text_cleaner(config["data"]["text"]["cleaners"][0])

    metadata_path = raw_dir / "metadata.csv"
    with open(metadata_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    metadata = []

    for line in tqdm(lines, desc="Preprocessing"):
        parts = line.strip().split("|")
        basename = parts[0]
        text = parts[2] if len(parts) > 2 else parts[1]

        wav_path = raw_dir / "wavs" / f"{basename}.wav"

        if not wav_path.exists():
            continue

        try:
            cleaned_text = text_cleaner(text)
            phone_sequence = phonemizer.text_to_sequence(cleaned_text)

            wav = audio_processor.load_wav(str(wav_path))
            mel = audio_processor.wav_to_mel(wav)
            pitch = audio_processor.extract_pitch(
                wav, use_log=config["audio"]["pitch"]["use_log_scale"]
            )
            energy = audio_processor.extract_energy(
                wav, use_log=config["audio"]["energy"]["use_log_scale"]
            )

            min_len = min(mel.shape[0], pitch.shape[0], energy.shape[0])
            mel = mel[:min_len]
            pitch = pitch[:min_len]
            energy = energy[:min_len]

            np.save(text_dir / f"{basename}.npy", phone_sequence)
            np.save(mel_dir / f"{basename}.npy", mel)
            np.save(pitch_dir / f"{basename}.npy", pitch)
            np.save(energy_dir / f"{basename}.npy", energy)

            metadata.append(
                {
                    "basename": basename,
                    "text": cleaned_text,
                    "text_len": len(phone_sequence),
                    "mel_len": mel.shape[0],
                }
            )

        except Exception as e:
            print(f"Error processing {basename}: {e}")
            continue

    train_split = int(len(metadata) * config["data"]["train_split"])
    train_metadata = metadata[:train_split]
    val_metadata = metadata[train_split:]

    with open(preprocessed_dir / "train.json", "w") as f:
        json.dump(train_metadata, f, indent=2)

    with open(preprocessed_dir / "val.json", "w") as f:
        json.dump(val_metadata, f, indent=2)

    print(f"Preprocessed {len(metadata)} samples")
    print(f"Train: {len(train_metadata)}, Val: {len(val_metadata)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    config = OmegaConf.to_container(config, resolve=True)

    preprocess_ljspeech(config)
