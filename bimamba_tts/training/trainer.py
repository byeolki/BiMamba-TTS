import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data import BiMambaTTSCollate
from ..losses import BiMambaTTSLoss
from ..models.bimamba_tts import BiMambaTTS
from ..utils import DiscordNotifier, Logger, load_checkpoint, save_checkpoint
from .optimizer import get_optimizer
from .scheduler import get_scheduler


class Trainer:
    def __init__(self, config, train_dataset, val_dataset):
        self.config = config
        self.device = torch.device(config["device"])

        self.model = BiMambaTTS(config).to(self.device)
        self.criterion = BiMambaTTSLoss(config)
        self.optimizer = get_optimizer(self.model, config)
        self.scheduler = get_scheduler(self.optimizer, config)

        collate_fn = BiMambaTTSCollate()

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config["train"]["batch_size"],
            shuffle=True,
            num_workers=config["train"]["num_workers"],
            pin_memory=config["train"]["pin_memory"],
            collate_fn=collate_fn,
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config["train"]["batch_size"],
            shuffle=False,
            num_workers=config["train"]["num_workers"],
            pin_memory=config["train"]["pin_memory"],
            collate_fn=collate_fn,
        )

        self.logger = Logger(config)
        self.scaler = (
            torch.amp.GradScaler("cuda") if config["train"]["mixed_precision"] else None
        )

        webhook_url = os.getenv("DISCORD_WEBHOOK_URL", "")
        discord_enabled = config.get("discord", {}).get("enabled", False)
        self.discord = DiscordNotifier(webhook_url if discord_enabled else None)
        self.notify_every_n_epochs = config.get("discord", {}).get(
            "notify_every_n_epochs", 100
        )

        self.global_step = 0
        self.current_epoch = 0

        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.patience_counter = 0
        self.early_stop_patience = (
            config["train"].get("early_stopping", {}).get("patience", 50)
        )
        self.early_stop_min_delta = (
            config["train"].get("early_stopping", {}).get("min_delta", 0.001)
        )

    def train_epoch(self):
        self.model.train()
        total_loss = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")

        for batch_idx, batch in enumerate(pbar):
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            self.optimizer.zero_grad()

            if self.scaler:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    predictions = self.model(
                        text=batch["texts"],
                        src_mask=batch["src_masks"],
                        mel_mask=batch["mel_masks"],
                        duration_target=batch["durations"],
                        pitch_target=batch["pitches"],
                        energy_target=batch["energies"],
                        max_len=batch["mels"].size(1),
                    )

                    losses = self.criterion(
                        predictions,
                        batch,
                        {
                            "src_masks": batch["src_masks"],
                            "mel_masks": batch["mel_masks"],
                        },
                    )

                self.scaler.scale(losses["total"]).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config["train"]["gradient_clip_val"]
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                predictions = self.model(
                    text=batch["texts"],
                    src_mask=batch["src_masks"],
                    mel_mask=batch["mel_masks"],
                    duration_target=batch["durations"],
                    pitch_target=batch["pitches"],
                    energy_target=batch["energies"],
                    max_len=batch["mels"].size(1),
                )

                losses = self.criterion(
                    predictions,
                    batch,
                    {"src_masks": batch["src_masks"], "mel_masks": batch["mel_masks"]},
                )

                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config["train"]["gradient_clip_val"]
                )
                self.optimizer.step()

            self.scheduler.step()

            total_loss += losses["total"].item()

            if self.global_step % self.config["train"]["log_every_n_steps"] == 0:
                metrics = {
                    "train/loss": losses["total"].item(),
                    "train/mel_loss": losses["mel"].item(),
                    "train/postnet_loss": losses["postnet"].item(),
                    "train/duration_loss": losses["duration"].item(),
                    "train/pitch_loss": losses["pitch"].item(),
                    "train/energy_loss": losses["energy"].item(),
                    "train/lr": self.optimizer.param_groups[0]["lr"],
                }
                self.logger.log(metrics, self.global_step)

            pbar.set_postfix({"loss": losses["total"].item()})
            self.global_step += 1

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0

        for batch in tqdm(self.val_loader, desc="Validation"):
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            predictions = self.model(
                text=batch["texts"],
                src_mask=batch["src_masks"],
                mel_mask=batch["mel_masks"],
                duration_target=batch["durations"],
                pitch_target=batch["pitches"],
                energy_target=batch["energies"],
                max_len=batch["mels"].size(1),
            )

            losses = self.criterion(
                predictions,
                batch,
                {"src_masks": batch["src_masks"], "mel_masks": batch["mel_masks"]},
            )

            total_loss += losses["total"].item()

        avg_loss = total_loss / len(self.val_loader)

        metrics = {"val/loss": avg_loss}
        self.logger.log(metrics, self.global_step)

        return avg_loss

    def train(self):
        self.discord.send_training_start(
            total_epochs=self.config["train"]["epochs"],
            batch_size=self.config["train"]["batch_size"],
            train_samples=len(self.train_loader.dataset),
            val_samples=len(self.val_loader.dataset),
            model_params={
                "d_model": self.config["model"]["encoder"]["d_model"],
                "n_layers": self.config["model"]["encoder"]["n_layers"],
                "lr": self.config["train"]["optimizer"]["lr"],
            },
        )

        for epoch in range(self.current_epoch, self.config["train"]["epochs"]):
            self.current_epoch = epoch

            train_loss = self.train_epoch()

            if epoch % self.config["train"]["validate_every_n_epochs"] == 0:
                val_loss = self.validate()
                print(
                    f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}"
                )

                if val_loss < self.best_val_loss - self.early_stop_min_delta:
                    self.best_val_loss = val_loss
                    self.best_epoch = epoch
                    self.patience_counter = 0
                    print(f"Val loss improved to {val_loss:.4f}")
                else:
                    self.patience_counter += 1
                    print(
                        f"No improvement. Patience: {self.patience_counter}/{self.early_stop_patience}"
                    )

                if epoch % self.notify_every_n_epochs == 0:
                    self.discord.send_training_stats(
                        epoch=epoch,
                        train_loss=train_loss,
                        val_loss=val_loss,
                        best_val_loss=self.best_val_loss,
                        patience_counter=self.patience_counter,
                        total_epochs=self.config["train"]["epochs"],
                        lr=self.optimizer.param_groups[0]["lr"],
                    )

                if self.patience_counter >= self.early_stop_patience:
                    print(f"Early stopping triggered at epoch {epoch}")
                    self._send_completion_notification(
                        epoch, train_loss, val_loss, "early_stopped"
                    )
                    break

            if epoch % self.config["train"]["save_every_n_epochs"] == 0:
                state = {
                    "epoch": epoch,
                    "global_step": self.global_step,
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "config": self.config,
                }
                save_checkpoint(
                    state,
                    self.config["paths"]["checkpoint_dir"],
                    self.global_step,
                )

        self._send_completion_notification(epoch, train_loss, val_loss, "completed")
        self.logger.finish()

    def _send_completion_notification(
        self, epoch: int, train_loss: float, val_loss: float, reason: str
    ):
        output_dir = Path(self.config["paths"]["output_dir"])
        audio_path = output_dir / "result.wav"

        self.discord.send_completion(
            total_epochs=epoch + 1,
            best_epoch=self.best_epoch,
            best_val_loss=self.best_val_loss,
            final_train_loss=train_loss,
            final_val_loss=val_loss,
            reason=reason,
            audio_path=audio_path if audio_path.exists() else None,
        )
