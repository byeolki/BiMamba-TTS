import argparse
import json
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))


def compute_statistics(config):
    preprocessed_dir = Path(config["paths"]["preprocessed_dir"])

    pitch_dir = preprocessed_dir / "pitch"
    energy_dir = preprocessed_dir / "energy"

    with open(preprocessed_dir / "train.json", "r") as f:
        metadata = json.load(f)

    pitches = []
    energies = []

    for item in tqdm(metadata, desc="Computing statistics"):
        basename = item["basename"]

        pitch = np.load(pitch_dir / f"{basename}.npy")
        energy = np.load(energy_dir / f"{basename}.npy")

        pitches.append(pitch[pitch != 0])
        energies.append(energy)

    pitches = np.concatenate(pitches)
    energies = np.concatenate(energies)

    stats = {
        "pitch_min": float(np.min(pitches)),
        "pitch_max": float(np.max(pitches)),
        "pitch_mean": float(np.mean(pitches)),
        "pitch_std": float(np.std(pitches)),
        "energy_min": float(np.min(energies)),
        "energy_max": float(np.max(energies)),
        "energy_mean": float(np.mean(energies)),
        "energy_std": float(np.std(energies)),
    }

    with open(preprocessed_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print("Statistics computed:")
    for k, v in stats.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    config = OmegaConf.to_container(config, resolve=True)

    compute_statistics(config)
