"""
    “What is the option worth today, according to Monte Carlo?”
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Any, Dict
import numpy as np
from pathlib import Path

from src.data.gbm import GBMParams, simulate_gbm_paths


@dataclass
class MCResult:
    price: float
    st: np.ndarray
    payoff: np.ndarray
    discounted_payoff: np.ndarray


def payoff_european(st: np.ndarray, k: float, option_type: str) -> np.ndarray:
    """
        compute European option payoff at maturity

        Args:
            st: maturity/terminal prices -> shape (n_paths)
            k: strike price
            option_type: "call" or "put

        returns:
            payoff ndarray with same shape as st
    """
    if k <= 0:
        raise ValueError(f"strike price k={k} must be > 0")

    opt = option_type.strip().lower()
    if opt not in {"call", "put"}:
        raise ValueError(f"option type {option_type} must be 'call' or 'put'")

    st = np.asarray(st, dtype=float)
    if opt == "call":
        return np.maximum(st - k, 0.0)
    else: #which will be put option
        return np.maximum(k - st, 0.0)

def mc_price_european(
        *,
        gbm_params: GBMParams,
        k: float,
        option_type: str,
        n_paths: int,
        seed: Optional[int] = None,
        r: Optional[float] = None,
) -> MCResult:
    """
        Monte Carlo price for a European option using GBM terminal prices

        returns: MCResult containing scaler price and arras for inspection
    """
    if n_paths <= 0:
        raise ValueError(f"n_paths must be > 0")
    if gbm_params.t <= 0:
        raise ValueError(f"t must be > 0")

    # simulate GBM paths
    paths = simulate_gbm_paths(
        params=gbm_params,
        n_paths=n_paths,
        seed=seed,
        return_log=False,
    )
    # Terminal prices
    st = paths[:, -1]

    # payoff at maturity
    payoff = payoff_european(st, k, option_type)

    # discounted back to T0
    rr = gbm_params.r if r is None else float(r)
    disc = np.exp(-rr * gbm_params.t)
    discounted_payoff = disc * payoff

    # monte carlo estimates
    price = float(discounted_payoff.mean())

    return MCResult(
        price=price,
        st=st,
        payoff=payoff,
        discounted_payoff=discounted_payoff,
    )

def save_mc_result_npz(
    *,
    result: MCResult,
    out_path: Path,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        price=result.price,
        st=result.st,
        payoff=result.payoff,
        discounted_payoff=result.discounted_payoff,
        meta=np.array(meta if meta is not None else {}, dtype=object),
    )