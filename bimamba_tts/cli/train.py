import argparse
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.append(str(Path(__file__).parent.parent.parent))

from bimamba_tts.data import LJSpeechDataset
from bimamba_tts.training import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train BiMamba-TTS")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Checkpoint path to resume from"
    )
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    config = OmegaConf.to_container(config, resolve=True)

    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config["seed"])

    preprocessed_dir = config["paths"]["preprocessed_dir"]

    train_dataset = LJSpeechDataset(preprocessed_dir, split="train")
    val_dataset = LJSpeechDataset(preprocessed_dir, split="val")

    trainer = Trainer(config, train_dataset, val_dataset)

    if args.checkpoint:
        from bimamba_tts.utils import load_checkpoint

        checkpoint = load_checkpoint(args.checkpoint, device=config["device"])
        trainer.model.load_state_dict(checkpoint["model"])
        trainer.optimizer.load_state_dict(checkpoint["optimizer"])
        trainer.current_epoch = checkpoint["epoch"] + 1
        trainer.global_step = checkpoint["global_step"]
        print(f"Resumed from epoch {checkpoint['epoch']}, step {trainer.global_step}")

    trainer.train()


if __name__ == "__main__":
    main()
