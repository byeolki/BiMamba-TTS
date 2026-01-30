class NoamScheduler:
    def __init__(self, optimizer, d_model=256, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.current_step = 0

    def step(self):
        self.current_step += 1
        lr = self.get_lr()

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def get_lr(self):
        step = self.current_step
        lr = self.d_model ** (-0.5) * min(
            step ** (-0.5), step * self.warmup_steps ** (-1.5)
        )
        return lr


def get_scheduler(optimizer, config):
    scheduler_config = config["train"]["scheduler"]

    if scheduler_config["name"] == "noam":
        scheduler = NoamScheduler(
            optimizer,
            d_model=config["model"]["encoder"]["d_model"],
            warmup_steps=scheduler_config["warmup_steps"],
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_config['name']}")

    return scheduler
