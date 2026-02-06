import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn

from src.physics.bs_pde_operator import bs_pde_residual


def norm_cdf(x: torch.Tensor) -> torch.Tensor:
    # Standard normal CDF using erf (no scipy needed)
    return 0.5 * (1.0 + torch.erf(x / torch.sqrt(torch.tensor(2.0, dtype=x.dtype, device=x.device))))


class BSClosedFormCall(nn.Module):
    """
    V(S,t) = BS call price for tau = T - t.
    Using torch ops so autodiff can compute derivatives.
    """
    def __init__(self, *, k: float, r: float, sigma: float, T: float):
        super().__init__()
        self.k = float(k)
        self.r = float(r)
        self.sigma = float(sigma)
        self.T = float(T)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x[:, 0:1]
        t = x[:, 1:2]
        tau = self.T - t

        # Avoid tau=0 inside log/div (we won't sample too close to T anyway)
        sigma = self.sigma
        r = self.r
        k = self.k

        sqrt_tau = torch.sqrt(tau)
        d1 = (torch.log(s / k) + (r + 0.5 * sigma**2) * tau) / (sigma * sqrt_tau)
        d2 = d1 - sigma * sqrt_tau

        return s * norm_cdf(d1) - k * torch.exp(-r * tau) * norm_cdf(d2)


def test_bs_pde_residual_near_zero_for_closed_form_call():
    torch.set_default_dtype(torch.float64)  # better numerical stability

    # parameters
    k, r, sigma, T = 100.0, 0.05, 0.2, 1.0

    # Sample interior points away from boundaries
    n = 200
    g = torch.Generator().manual_seed(123)

    s = (1.0 + 299.0 * torch.rand((n, 1), generator=g))  # (1,300) avoids 0
    t = (0.02 + 0.96 * torch.rand((n, 1), generator=g))  # (0.02,0.98) avoids 0 and T

    # Must require gradients for autodiff
    s.requires_grad_(True)
    t.requires_grad_(True)

    model = BSClosedFormCall(k=k, r=r, sigma=sigma, T=T)

    res = bs_pde_residual(model=model, s=s, t=t, r=r, sigma=sigma)
    max_abs = res.abs().max().item()
    mean_abs = res.abs().mean().item()

    # Residual should be ~0 (allow small numerical noise)
    assert max_abs < 1e-6, f"Max |residual| too large: {max_abs}"
    assert mean_abs < 1e-7, f"Mean |residual| too large: {mean_abs}"
