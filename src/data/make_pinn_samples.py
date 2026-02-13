"""
    Blueprint for where the network must satisfy the Black Scholes laws in (S, t) domain


        S_max ────────────────────────────────
          |   boundary (high S)
          |
          |   •  •   •    interior points
          |    •    •   •
          |  •   •    •
          |
          |   boundary (low S)
        S=0 ────────────────────────────────
              t=0                    t=T
                        terminal payoff line

    Three types of constraints
    Interior points
        → “At these points, the PDE must be zero”

    Terminal points
        → “At maturity, price equals payoff”

    Boundary points
        → “At extreme prices, behavior is known”

"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional, Dict
import numpy as np

from src.data.make_mc_dataset import payoff_european

@dataclass
class TerminalSet:
    s: np.ndarray   #shape (n,1)
    t: np.ndarray   #shape (n,1) all = T
    v: np.ndarray   #shape (n,1) payoff labels

@dataclass
class InteriorSet:
    s: np.ndarray
    t: np.ndarray

@dataclass(frozen=True)
class BoundarySet:
    s: np.ndarray   # (n, 1)
    t: np.ndarray   # (n, 1)
    v: np.ndarray   # (n, 1)


@dataclass(frozen=True)
class PINNDataset:
    terminal: TerminalSet
    interior: InteriorSet
    boundary: Dict[str, BoundarySet]

def sample_terminal_points(
        *,
        n: int,
        t_maturity: float,
        k: float,
        option_type: str,
        s_max: Optional[float] = None,
        seed: Optional[int] = None,
        sampler: Literal["uniform"] = "uniform",
) -> TerminalSet:
    """
        At the terminal t = T, the option price equals the payoff
        so,
            V(S,T) = payoff(S)

        Return:
            TerminalSet with s, t, v arrays shaped (n,1)
    """
    if n <=0 :
        raise ValueError(f"n must be >= 0")
    if  t_maturity <= 0:
        raise ValueError(f"t_maturity must be >= 0")
    if k <= 0:
        raise ValueError(f"k must be >= 0")
    
    s_hi = float(s_max) if s_max is not None else 3.0 * float(k)
    
    if s_hi <= 0:
        raise ValueError(f"s_hi must be >= 0")
    if sampler != "uniform":
        raise ValueError(f"Unsupported sampler {sampler!r}. Only uniform sampler is supported")
    
    rng = np.random.default_rng(seed)
    # sample  S in [0, s_hi]
    s = rng.uniform(0.0, s_hi, size=(n,))
    
    # set t = T for all points
    t = np.full(shape=(n,), fill_value=float(t_maturity), dtype=float)
    
    # compute payoff labels at maturity
    v = payoff_european(st=s, k=float(k), option_type=option_type)
    
    # reshape to (n,1) for consistent NN input
    s = s.reshape(-1,1)
    t = t.reshape(-1,1)
    v = v.reshape(-1,1)
    
    return TerminalSet(s=s, t=t, v=v)

def sample_interior_points(
        *,
        n: int,
        t_maturity: float,
        k: float,
        s_max: Optional[float] = None,
        seed: Optional[int] = None,
        sampler: Literal["uniform", "mixture_k"] = "uniform",
        eps_t: float = 1e-6,
        eps_s: float = 1e-6,
        mix_frac: float = 0.5,          # fraction near K
        k_std_frac: float = 0.4,        # std as fraction of K
) -> InteriorSet:
    ...
    s_hi = float(s_max) if s_max is not None else 3.0 * float(k)
    rng = np.random.default_rng(seed)

    # time is always uniform in (0, T)
    t = rng.uniform(eps_t, float(t_maturity) - eps_t, size=(n,))

    if sampler == "uniform":
        s = rng.uniform(eps_s, s_hi - eps_s, size=(n,))

    elif sampler == "mixture_k":
        n_k = int(round(n * mix_frac))
        n_u = n - n_k

        # uniform component
        s_u = rng.uniform(eps_s, s_hi - eps_s, size=(n_u,))

        # near-strike component (normal around K)
        mu = float(k)
        std = max(1e-12, float(k_std_frac) * float(k))
        s_k = rng.normal(loc=mu, scale=std, size=(n_k,))

        # clip into (0, Smax) open interval
        s_k = np.clip(s_k, eps_s, s_hi - eps_s)

        s = np.concatenate([s_u, s_k], axis=0)
        rng.shuffle(s)

    else:
        raise ValueError(f"Unsupported sampler {sampler!r}. Use 'uniform' or 'mixture_k'.")

    return InteriorSet(s=s.reshape(-1, 1), t=t.reshape(-1, 1))


def boundary_value(
        *,
        s: np.ndarray,
        t: np.ndarray,
        k: float,
        r: float,
        t_maturity: float,
        option_type: str,
        side: Literal["low", "high"],
) -> np.ndarray:
    """
        Compute boundary condition values for European options on S boundaries

        low side:
            call : V(0, t) = 0
            put : V(0, t) = k * exp (-r*(T-t))

        high side (S = S_max approximation):
            call : V(S ,t) = S - k * exp (-r*(T-t))
            put : V(S, t) = 0
    """

    opt = option_type.strip().lower()
    if opt not in ["call", "put"]:
        raise ValueError(f"option_type must be 'call' or 'put', got {opt!r}")

    tau = float(t_maturity) - t   # time to maturity

    if np.any(tau<0):
        raise ValueError(f"t contains value > T (negative time-to-maturity!!)")

    if side == "low":
        if opt == "call":
            return np.zeros_like(t, dtype=float)
        else: #put option
            return float(k) * np.exp(-float(r) * tau)
    elif side == "high":
        if opt == "call":
            return s - float(k) * np.exp(-float(r) * tau)
        else: # put option
            return np.zeros_like(t, dtype=float)
    else:
        raise ValueError(f"side must be 'low' or 'high', got {side!r}")


def sample_boundary_points(
        *,
        n: int,
        t_maturity: float,
        k: float,
        r: float,
        option_type: str,
        s_max: Optional[float] = None,
        seed: Optional[int] = None,
        sampler: Literal["uniform"] = "uniform",
        eps_t: float = 1e-6,
) -> Dict[str, BoundarySet]:
    """
        Sample boundary points on S=0 (low) and S=S_max (high)

        returns:
            dict with keys: (low and high)
    """

    if n <= 0:
        raise ValueError(f"n must be >= 0")
    if t_maturity <= 0:
        raise ValueError(f"t_maturity must be >= 0")
    if k <= 0:
        raise ValueError(f"k must be >= 0")

    s_hi = float(s_max) if s_max is not None else 3.0 * float(k)
    if s_hi <= 0:
        raise ValueError(f"s_hi must be >= 0")
    if sampler != "uniform":
        raise ValueError(f"Unsupported sampler {sampler!r}. Only uniform sampler is supported")

    rng = np.random.default_rng(seed)

    # sample times along the boundary (exclude endpoints)
    t = rng.uniform(eps_t, float(t_maturity - eps_t), size=(n,)).reshape(-1,1)

    # low boundary : S=0
    s_low = np.zeros((n,1), dtype=float)
    v_low = boundary_value(
        s=s_low,
        t=t,
        k=k,
        r=r,
        t_maturity=t_maturity,
        option_type=option_type,
        side="low"
    )

    # high boundary: S=S_max
    s_high = np.full((n,1),  fill_value=float(s_hi), dtype=float)
    v_high = boundary_value(
        s=s_high,
        t=t,
        k=k,
        r=r,
        t_maturity=t_maturity,
        option_type=option_type,
        side="high"
    )

    return {
        "low": BoundarySet(s=s_low, t=t, v=v_low.reshape(-1,1)),
        "high": BoundarySet(s=s_high, t=t, v=v_high.reshape(-1,1))
    }


def make_pinn_dataset(
        *,
        n_terminal: int,
        n_interior: int,
        n_boundary: int,
        t_maturity: float,
        k: float,
        r: float,
        option_type: str,
        s_max: float | None = None,
        seed: int | None = None,
) -> PINNDataset:
    """
        Create the full PINN Dataset (constraint blueprint)
            - terminal payoff points
            - interior collocation points
            - boundary condition points
    """
    terminal = sample_terminal_points(
        n=n_terminal,
        t_maturity=t_maturity,
        k=k,
        option_type=option_type,
        s_max=s_max,
        seed=seed,
    )

    interior = sample_interior_points(
        n=n_interior,
        t_maturity=t_maturity,
        k=k,
        s_max=s_max,
        seed=None if seed is None else seed + 1,
        sampler="mixture_k",
    )

    boundary = sample_boundary_points(
        n=n_boundary,
        t_maturity=t_maturity,
        k=k,
        r=r,
        option_type=option_type,
        s_max=s_max,
        seed=None if seed is None else seed + 2,
    )

    return PINNDataset(
        terminal=terminal,
        interior=interior,
        boundary=boundary,
    )

def make_pinn_dataset_from_cfg(cfg) -> PINNDataset:
    """
        Create the PINN Dataset (constraint blueprint) from config
    """

    data_cfg = cfg.data
    option_cfg = cfg.data.option
    train_cfg = cfg.train

    return make_pinn_dataset(
        n_terminal=train_cfg.pinn.n_terminal,
        n_interior=train_cfg.pinn.n_interior,
        n_boundary=train_cfg.pinn.n_boundary,
        t_maturity=data_cfg.t,
        k=data_cfg.k,
        r=data_cfg.r,
        option_type=option_cfg["type"] if isinstance(option_cfg, dict) else option_cfg.type,
        s_max=getattr(option_cfg, "s_max", None),
        seed = data_cfg.seed,
    )
