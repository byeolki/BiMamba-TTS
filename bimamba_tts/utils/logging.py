from pathlib import Path

import torch
import wandb
from dotenv import load_dotenv

load_dotenv()


class Logger:
    def __init__(self, config, enabled=True):
        self.enabled = enabled and config.get("wandb", {}).get("enabled", False)

        if self.enabled:
            wandb.init(
                project=config["wandb"]["project"],
                entity=config["wandb"].get("entity", None),
                config=dict(config),
                name=config.get("experiment_name", "bimamba-tts"),
            )

    def log(self, metrics, step):
        if self.enabled:
            wandb.log(metrics, step=step)

    def log_audio(self, tag, audio, step, sample_rate=22050):
        if self.enabled:
            wandb.log({tag: wandb.Audio(audio, sample_rate=sample_rate)}, step=step)

    def log_image(self, tag, image, step):
        if self.enabled:
            wandb.log({tag: wandb.Image(image)}, step=step)

    def finish(self):
        if self.enabled:
            wandb.finish()


def save_checkpoint(state, checkpoint_dir, step):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / f"checkpoint_{step}.pt"
    torch.save(state, checkpoint_path)

    latest_path = checkpoint_dir / "latest.pt"
    torch.save(state, latest_path)


def load_checkpoint(checkpoint_path, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    return checkpoint
