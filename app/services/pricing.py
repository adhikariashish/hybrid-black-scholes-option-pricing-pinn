# app/services/pricing.py
from __future__ import annotations

import math
from typing import Iterable, Optional

import numpy as np
import torch

from app.services.loaders import load_model_cached


# -------------------------
# Black–Scholes closed form
# -------------------------
def _norm_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def bs_price(
    *,
    s: np.ndarray,
    k: float,
    r: float,
    sigma: float,
    t: float,
    T: float,
    option_type: str,
) -> np.ndarray:
    """
    European Black–Scholes price at time t for maturity T.
    s: spot array
    """
    s = np.asarray(s, dtype=float)
    opt = option_type.lower().strip()
    if opt not in ("call", "put"):
        raise ValueError(f"option_type must be call/put, got {option_type!r}")

    tau = max(T - t, 0.0)

    # Handle tau ~ 0: payoff at maturity
    if tau <= 1e-12:
        if opt == "call":
            return np.maximum(s - k, 0.0)
        return np.maximum(k - s, 0.0)

    # Avoid divide-by-zero
    sigma = float(sigma)
    if sigma <= 0:
        # deterministic forward payoff with discount
        disc = math.exp(-r * tau)
        if opt == "call":
            return disc * np.maximum(s - k, 0.0)
        return disc * np.maximum(k - s, 0.0)

    sqrt_tau = math.sqrt(tau)
    d1 = (np.log(np.maximum(s, 1e-300) / k) + (r + 0.5 * sigma * sigma) * tau) / (sigma * sqrt_tau)
    d2 = d1 - sigma * sqrt_tau

    Nd1 = _norm_cdf(d1)
    Nd2 = _norm_cdf(d2)

    disc = math.exp(-r * tau)

    if opt == "call":
        return s * Nd1 - k * disc * Nd2

    # put
    Nmd1 = _norm_cdf(-d1)
    Nmd2 = _norm_cdf(-d2)
    return k * disc * Nmd2 - s * Nmd1


# -------------------------
# PINN inference
# -------------------------
def pinn_price(
    *,
    run_dir: str,
    option_type: str,
    s: np.ndarray,
    t: float,
) -> np.ndarray:
    """
    Uses trained model with normalized inputs:
      s_norm = s / s_max
      t_norm = t / T
    Assumes model outputs V directly (not normalized).
    """
    model, cfg = load_model_cached(run_dir, option_type)

    # Pull normalization constants from config snapshot
    T = float(cfg["data"]["t"])
    s_max = float(cfg["data"]["option"]["s_max"])

    s = np.asarray(s, dtype=float)
    s_norm = s / s_max
    t_norm = float(t) / T if T > 0 else 0.0

    device = next(model.parameters()).device
    dtype = torch.get_default_dtype()

    s_t = torch.tensor(s_norm.reshape(-1, 1), device=device, dtype=dtype)
    t_t = torch.full((s_t.shape[0], 1), float(t_norm), device=device, dtype=dtype)

    x = torch.cat([s_t, t_t], dim=1)

    with torch.no_grad():
        y = model(x).detach().cpu().numpy().reshape(-1)

    # ensure non-negative
    return np.maximum(y, 0.0)


# -------------------------
# Convenience wrappers used by UI
# -------------------------
def bs_price_curve(params, s: Iterable[float], t_grid: Optional[np.ndarray] = None):
    s = np.asarray(list(s), dtype=float)

    if t_grid is None:
        t_eval = float(getattr(params, "t_eval", 0.0))
        return bs_price(
            s=s,
            k=float(params.k),
            r=float(params.r),
            sigma=float(params.sigma),
            t=t_eval,
            T=float(params.t_maturity),
            option_type=str(params.option_type),
        )

    # surface: return Z[t, s]
    t_grid = np.asarray(t_grid, dtype=float)
    Z = np.zeros((len(t_grid), len(s)), dtype=float)
    for i, tt in enumerate(t_grid):
        Z[i, :] = bs_price(
            s=s,
            k=float(params.k),
            r=float(params.r),
            sigma=float(params.sigma),
            t=float(tt),
            T=float(params.t_maturity),
            option_type=str(params.option_type),
        )
    return Z


def pinn_price_curve(params, s: Iterable[float], t_grid: Optional[np.ndarray] = None):
    s = np.asarray(list(s), dtype=float)

    if t_grid is None:
        t_eval = float(getattr(params, "t_eval", 0.0))
        return pinn_price(run_dir=str(params.run_dir), option_type=str(params.option_type), s=s, t=t_eval)

    t_grid = np.asarray(t_grid, dtype=float)
    Z = np.zeros((len(t_grid), len(s)), dtype=float)
    for i, tt in enumerate(t_grid):
        Z[i, :] = pinn_price(run_dir=str(params.run_dir), option_type=str(params.option_type), s=s, t=float(tt))
    return Z
