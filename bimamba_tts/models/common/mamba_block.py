import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class MambaBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True,
        )

        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)

        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        A = repeat(torch.arange(1, d_state + 1), "n -> d n", d=self.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        residual = x
        x = self.norm(x)

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = rearrange(x, "b l d -> b d l")
        x = self.conv1d(x)[:, :, :seq_len]
        x = rearrange(x, "b d l -> b l d")

        x = F.silu(x)

        x_dbl = self.x_proj(x)
        delta, B, C = (
            x_dbl[:, :, :1],
            x_dbl[:, :, 1 : self.d_state + 1],
            x_dbl[:, :, self.d_state + 1 :],
        )

        delta = F.softplus(self.dt_proj(delta))

        y = self.selective_scan(x, delta, B, C)

        y = y * F.silu(z)

        output = self.out_proj(y)
        output = self.dropout(output)

        return output + residual

    def selective_scan(
        self,
        x: torch.Tensor,
        delta: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        batch, seq_len, d_inner = x.shape

        A = -torch.exp(self.A_log.float())

        deltaA = torch.exp(delta.unsqueeze(-1) * A)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)

        h = torch.zeros(batch, d_inner, self.d_state, device=x.device, dtype=x.dtype)
        ys = []

        for i in range(seq_len):
            h = deltaA[:, i] * h + deltaB[:, i] * x[:, i, :, None]
            y = (h * C[:, i, None, :]).sum(dim=-1)
            ys.append(y)

        y = torch.stack(ys, dim=1)

        y = y + x * self.D

        return y
