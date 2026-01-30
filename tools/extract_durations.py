import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from bimamba_tts.utils import MontrealForcedAligner


def prepare_mfa_corpus(config):
    raw_dir = Path(config["paths"]["raw_dir"])
    preprocessed_dir = Path(config["paths"]["preprocessed_dir"])
    mfa_corpus_dir = preprocessed_dir / "mfa_corpus"
    mfa_corpus_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = raw_dir / "metadata.csv"
    with open(metadata_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in tqdm(lines, desc="Preparing MFA corpus"):
        parts = line.strip().split("|")
        basename = parts[0]
        text = parts[2] if len(parts) > 2 else parts[1]

        wav_src = raw_dir / "wavs" / f"{basename}.wav"
        wav_dst = mfa_corpus_dir / f"{basename}.wav"
        txt_dst = mfa_corpus_dir / f"{basename}.txt"

        if wav_src.exists():
            os.system(f"cp {wav_src} {wav_dst}")

            with open(txt_dst, "w", encoding="utf-8") as f:
                f.write(text)

    print(f"MFA corpus prepared at {mfa_corpus_dir}")


def extract_durations_from_textgrids(config):
    preprocessed_dir = Path(config["paths"]["preprocessed_dir"])
    duration_dir = preprocessed_dir / "duration"
    duration_dir.mkdir(parents=True, exist_ok=True)

    aligned_dir = preprocessed_dir / "mfa_aligned"

    if not aligned_dir.exists():
        print(f"Aligned directory not found: {aligned_dir}")
        print("Please run MFA alignment first")
        return

    mfa = MontrealForcedAligner(
        corpus_dir=str(preprocessed_dir / "mfa_corpus"),
        dict_path="english_us_arpa",
        output_dir=str(aligned_dir),
    )

    textgrid_files = list(aligned_dir.glob("*.TextGrid"))

    for tg_path in tqdm(textgrid_files, desc="Extracting durations"):
        basename = tg_path.stem

        try:
            durations = mfa.get_phoneme_durations(
                str(tg_path),
                config["audio"]["hop_length"],
                config["audio"]["sampling_rate"],
            )

            np.save(duration_dir / f"{basename}.npy", durations)

        except Exception as e:
            print(f"Error processing {basename}: {e}")
            continue

    print(f"Extracted durations for {len(textgrid_files)} files")


def run_mfa_alignment(config):
    preprocessed_dir = Path(config["paths"]["preprocessed_dir"])

    mfa = MontrealForcedAligner(
        corpus_dir=str(preprocessed_dir / "mfa_corpus"),
        dict_path="english_us_arpa",
        acoustic_model="english_us_arpa",
        output_dir=str(preprocessed_dir / "mfa_aligned"),
    )

    print("Running MFA alignment...")
    mfa.align()
    print("MFA alignment completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument(
        "--step", type=str, choices=["prepare", "align", "extract"], required=True
    )
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    config = OmegaConf.to_container(config, resolve=True)

    if args.step == "prepare":
        prepare_mfa_corpus(config)
    elif args.step == "align":
        run_mfa_alignment(config)
    elif args.step == "extract":
        extract_durations_from_textgrids(config)
