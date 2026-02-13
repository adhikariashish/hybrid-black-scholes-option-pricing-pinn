from __future__ import annotations
import torch

# interior collocation PDE Loss

from src.physics.bs_pde_operator import bs_pde_residual

def physics_loss(
        *,
        model,
        s: torch.Tensor,
        t: torch.Tensor,
        r: float,
        sigma: float,
        T: float,
) -> torch.Tensor:
    res = bs_pde_residual(model=model, s=s, t=t, r=r, sigma=sigma, T=T)
    return torch.mean(res ** 2)