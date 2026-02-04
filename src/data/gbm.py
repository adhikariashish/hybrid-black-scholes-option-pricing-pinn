"""
    We assume the stock price Sₜ follows Geometric Brownian Motion:
                dSₜ = r Sₜ dt + σ Sₜ dWₜ
    Where:
        * r = risk-free rate
        * σ = volatility
        * Wₜ = Brownian motion (random noise)
    This is stochastic differential equation (SDE) described as:
        over an infinitesimal time dₜ -> the price changes due to
            - a smooth drift -> r Sₜ dt
            - a random shock -> σ Sₜ dWₜ
    here, the dₜ is infinitesimally small and dWₜ is not a normal variable rather a Brownian increment

    As such we solve the SDE,
        we get,
            Sₜ₊Δₜ = Sₜ exp((r - ½σ²)Δt + σ√Δt Z)

        where,
        Sₜ = stock price at time t
        r = risk-free rate
        σ = volatility
        Δt = time step size
        Z ~ N(0,1) = standard normal random variable

"""

from __future__ import annotations
from dataclasses import dataclass

from typing import Optional

import numpy as np

@dataclass(frozen=True)
class GBMParams:
    s0: float
    r : float
    sigma: float
    t:float
    n_steps: int

def simulate_gbm_paths(
        *,
        params: GBMParams,
        n_paths: int,
        seed: Optional[int] = None,
        return_log: bool = False,
)  -> np.ndarray:
    """
        simulate GBM paths under risk-neutral drift

    """
    if n_paths <= 0:
        raise ValueError(f"n_paths must be > 0, got {n_paths}")
    if params.n_steps <= 0:
        raise ValueError(f"n_steps must be > 0, got {params.n_steps}")
    if params.t <= 0:
        raise ValueError(f"t must be > 0, got {params.t}")
    if params.sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {params.sigma}")

    dt = params.t / params.n_steps
    rng = np.random.default_rng(seed)

    # Z ~ N(0,1), one per path per time step
    z = rng.standard_normal(size=(n_paths, params.n_steps))

    drift = (params.r - 0.5 * params.sigma ** 2) * dt
    diffusion = params.sigma * np.sqrt(dt) * z

    # log S_{t+dt} = log S_t + drift + diffusion
    log_paths = np.empty((n_paths, params.n_steps + 1), dtype=float)
    log_paths[:, 0] = np.log(params.s0)

    # cumulative sum across time (axis=1)
    log_paths[:, 1:] = log_paths[:, [0]] + np.cumsum(drift + diffusion, axis=1)

    if return_log:
        return log_paths

    return np.exp(log_paths)
