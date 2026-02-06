from __future__ import annotations

import torch

def mse_value_loss(
        *,
        model,
        s: torch.Tensor,
        t: torch.Tensor,
        v_true: torch.Tensor,
) -> torch.Tensor:
    x = torch.cat([s, t], dim=1)
    v_pred = model(x)
    return torch.mean((v_pred - v_true) ** 2)