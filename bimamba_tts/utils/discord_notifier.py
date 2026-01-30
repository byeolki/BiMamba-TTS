import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests


class DiscordNotifier:
    def __init__(self, webhook_url: Optional[str] = None):
        self.webhook_url = webhook_url
        self.enabled = webhook_url is not None and len(webhook_url) > 0

    def send_training_start(
        self,
        total_epochs: int,
        batch_size: int,
        train_samples: int,
        val_samples: int,
        model_params: dict,
    ):
        if not self.enabled:
            return

        embed = {
            "title": "Training Started!",
            "color": 3066993,
            "fields": [
                {"name": "Total Epochs", "value": str(total_epochs), "inline": True},
                {"name": "Batch Size", "value": str(batch_size), "inline": True},
                {
                    "name": "Train Samples",
                    "value": str(train_samples),
                    "inline": True,
                },
                {"name": "Val Samples", "value": str(val_samples), "inline": True},
                {
                    "name": "Model Dimensions",
                    "value": f"d_model={model_params.get('d_model', 'N/A')}, layers={model_params.get('n_layers', 'N/A')}",
                    "inline": False,
                },
                {
                    "name": "Learning Rate",
                    "value": f"{model_params.get('lr', 'N/A')}",
                    "inline": True,
                },
            ],
            "timestamp": datetime.utcnow().isoformat(),
        }

        payload = {"embeds": [embed]}

        try:
            response = requests.post(self.webhook_url, json=payload, timeout=10)
            response.raise_for_status()
        except Exception as e:
            print(f"Failed to send Discord start notification: {e}")

    def send_training_stats(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        best_val_loss: float,
        patience_counter: int,
        total_epochs: int,
        lr: float,
    ):
        if not self.enabled:
            return

        embed = {
            "title": f"Training Progress - Epoch {epoch}/{total_epochs}",
            "color": 3447003,
            "fields": [
                {"name": "Train Loss", "value": f"{train_loss:.4f}", "inline": True},
                {"name": "Val Loss", "value": f"{val_loss:.4f}", "inline": True},
                {
                    "name": "Best Val Loss",
                    "value": f"{best_val_loss:.4f}",
                    "inline": True,
                },
                {
                    "name": "Learning Rate",
                    "value": f"{lr:.6f}",
                    "inline": True,
                },
                {
                    "name": "Patience",
                    "value": f"{patience_counter}/20",
                    "inline": True,
                },
            ],
            "timestamp": datetime.utcnow().isoformat(),
        }

        payload = {"embeds": [embed]}

        try:
            response = requests.post(self.webhook_url, json=payload, timeout=10)
            response.raise_for_status()
        except Exception as e:
            print(f"Failed to send Discord notification: {e}")

    def send_completion(
        self,
        total_epochs: int,
        best_epoch: int,
        best_val_loss: float,
        final_train_loss: float,
        final_val_loss: float,
        reason: str,
        audio_path: Optional[Path] = None,
    ):
        if not self.enabled:
            return

        color = 5763719 if reason == "completed" else 15844367

        embed = {
            "title": f"Training {reason.capitalize()}!",
            "color": color,
            "fields": [
                {"name": "Total Epochs", "value": str(total_epochs), "inline": True},
                {"name": "Best Epoch", "value": str(best_epoch), "inline": True},
                {
                    "name": "Best Val Loss",
                    "value": f"{best_val_loss:.4f}",
                    "inline": True,
                },
                {
                    "name": "Final Train Loss",
                    "value": f"{final_train_loss:.4f}",
                    "inline": True,
                },
                {
                    "name": "Final Val Loss",
                    "value": f"{final_val_loss:.4f}",
                    "inline": True,
                },
                {"name": "Reason", "value": reason, "inline": False},
            ],
            "timestamp": datetime.utcnow().isoformat(),
        }

        payload = {"embeds": [embed]}

        try:
            if audio_path and audio_path.exists():
                with open(audio_path, "rb") as f:
                    files = {"file": (audio_path.name, f, "audio/wav")}
                    response = requests.post(
                        self.webhook_url,
                        data={"payload_json": json.dumps(payload)},
                        files=files,
                        timeout=30,
                    )
            else:
                response = requests.post(self.webhook_url, json=payload, timeout=10)

            response.raise_for_status()
        except Exception as e:
            print(f"Failed to send Discord completion notification: {e}")
