import torch
from torch.optim import AdamW


def get_optimizer(model, config):
    optimizer_config = config["train"]["optimizer"]

    if optimizer_config["name"] == "AdamW":
        optimizer = AdamW(
            model.parameters(),
            lr=optimizer_config["lr"],
            betas=optimizer_config["betas"],
            eps=optimizer_config["eps"],
            weight_decay=optimizer_config["weight_decay"],
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_config['name']}")

    return optimizer
