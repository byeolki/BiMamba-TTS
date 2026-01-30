import torch
import torch.nn as nn


class LengthRegulator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, x: torch.Tensor, duration: torch.Tensor, max_len: int = None
    ) -> torch.Tensor:
        output = []

        for batch_idx in range(x.size(0)):
            expanded = []
            for frame_idx, d in enumerate(duration[batch_idx]):
                d = int(d.item())
                if d > 0:
                    expanded.append(x[batch_idx, frame_idx].unsqueeze(0).expand(d, -1))

            if len(expanded) > 0:
                expanded = torch.cat(expanded, dim=0)
            else:
                expanded = torch.zeros(1, x.size(-1), device=x.device)

            output.append(expanded)

        if max_len is None:
            max_len = max([out.size(0) for out in output])

        output_padded = []
        for out in output:
            if out.size(0) < max_len:
                pad = torch.zeros(max_len - out.size(0), out.size(1), device=out.device)
                out = torch.cat([out, pad], dim=0)
            output_padded.append(out.unsqueeze(0))

        return torch.cat(output_padded, dim=0)
