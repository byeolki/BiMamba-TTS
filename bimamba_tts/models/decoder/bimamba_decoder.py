import torch
import torch.nn as nn

from ..common import BiMambaBlock, PositionalEncoding


class BiMambaDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layers: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        n_mel_channels: int = 80,
        dropout: float = 0.2,
        max_seq_len: int = 5000,
    ):
        super().__init__()

        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [
                BiMambaBlock(d_model, d_state, d_conv, expand, dropout)
                for _ in range(n_layers)
            ]
        )

        self.mel_linear = nn.Linear(d_model, n_mel_channels)
        nn.init.xavier_uniform_(self.mel_linear.weight)
        nn.init.constant_(self.mel_linear.bias, 0.0)

        self.postnet = nn.Sequential(
            nn.Conv1d(n_mel_channels, 512, kernel_size=5, padding=2),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Conv1d(512, 512, kernel_size=5, padding=2),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Conv1d(512, 512, kernel_size=5, padding=2),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Conv1d(512, 512, kernel_size=5, padding=2),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Conv1d(512, n_mel_channels, kernel_size=5, padding=2),
            nn.Dropout(0.5),
        )

        for m in self.postnet.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> tuple:
        x = self.pos_encoding(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, mask)

        mel_out = self.mel_linear(x)

        mel_postnet = self.postnet(mel_out.transpose(1, 2))
        mel_postnet = mel_postnet.transpose(1, 2)
        mel_postnet = mel_out + mel_postnet

        return mel_out, mel_postnet
