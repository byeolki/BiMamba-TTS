import torch
import torch.nn as nn


class DurationPredictor(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layers: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(n_layers):
            self.convs.append(
                nn.Conv1d(d_model, d_model, kernel_size, padding=(kernel_size - 1) // 2)
            )
            self.norms.append(nn.LayerNorm(d_model))

        self.linear = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)

        for conv, norm in zip(self.convs, self.norms):
            x = conv(x)
            x = x.transpose(1, 2)
            x = norm(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = x.transpose(1, 2)

        x = x.transpose(1, 2)
        x = self.linear(x)
        x = x.squeeze(-1)

        return x
