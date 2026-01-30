import argparse
import sys
from pathlib import Path

from omegaconf import OmegaConf

sys.path.append(str(Path(__file__).parent.parent.parent))

from tools.compute_statistics import compute_statistics
from tools.extract_durations import (
    extract_durations_from_textgrids,
    prepare_mfa_corpus,
    run_mfa_alignment,
)
from tools.preprocess_dataset import preprocess_ljspeech


def main():
    parser = argparse.ArgumentParser(description="Preprocess LJSpeech dataset")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument(
        "--stage", type=int, default=0, help="0: all, 1: audio, 2: mfa, 3: stats"
    )
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    config = OmegaConf.to_container(config, resolve=True)

    if args.stage == 0 or args.stage == 1:
        print("Stage 1: Preprocessing audio and text...")
        preprocess_ljspeech(config)

    if args.stage == 0 or args.stage == 2:
        print("\nStage 2: Montreal Forced Alignment...")
        print("Step 1: Preparing MFA corpus...")
        prepare_mfa_corpus(config)

        print("Step 2: Running MFA alignment...")
        run_mfa_alignment(config)

        print("Step 3: Extracting durations from TextGrids...")
        extract_durations_from_textgrids(config)

    if args.stage == 0 or args.stage == 3:
        print("\nStage 3: Computing statistics...")
        compute_statistics(config)

    print("\nPreprocessing completed!")


if __name__ == "__main__":
    main()
