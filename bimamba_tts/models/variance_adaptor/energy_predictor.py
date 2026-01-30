import torch
import torch.nn as nn


class EnergyPredictor(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layers: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.3,
        n_bins: int = 256,
        energy_min: float = -4.5,
        energy_max: float = 6.0,
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

        self.energy_bins = nn.Parameter(
            torch.linspace(energy_min, energy_max, n_bins - 1), requires_grad=False
        )
        self.energy_embedding = nn.Embedding(n_bins, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)

        for conv, norm in zip(self.convs, self.norms):
            residual = x
            x = conv(x)
            x = x.transpose(1, 2)
            x = norm(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = x.transpose(1, 2)
            x = x + residual

        x = x.transpose(1, 2)
        x = self.linear(x)
        x = x.squeeze(-1)

        return x

    def get_energy_embedding(self, energy: torch.Tensor) -> torch.Tensor:
        energy_idx = torch.bucketize(energy, self.energy_bins)
        return self.energy_embedding(energy_idx)
