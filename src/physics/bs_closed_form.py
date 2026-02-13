from __future__ import annotations

import math
from typing import Literal, Tuple, Dict

import numpy as np

OptionType = Literal["call", "put"]


# -----------------------------
# Normal CDF / PDF (no scipy)
# -----------------------------
_SQRT_2PI = math.sqrt(2.0 * math.pi)
_SQRT_2 = math.sqrt(2.0)


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    # Φ(x) = 0.5 * (1 + erf(x/sqrt(2)))
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / _SQRT_2))


def _norm_pdf(x: np.ndarray) -> np.ndarray:
    # φ(x) = exp(-0.5 x^2) / sqrt(2π)
    return np.exp(-0.5 * x * x) / _SQRT_2PI


def bs_d1_d2(S: np.ndarray, K: float, r: float, sigma: float, tau: float) -> Tuple[np.ndarray, np.ndarray]:
    if K <= 0:
        raise ValueError(f"K must be > 0, got {K}")
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}")
    if tau <= 0:
        raise ValueError(f"tau must be > 0, got {tau}")

    S = np.asarray(S, dtype=np.float64)

    # Avoid log(0) / divide by 0; caller masks S<=0 anyway.
    d1 = (np.log(S / K) + (r + 0.5 * sigma * sigma) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    return d1, d2


def bs_european_price(
    *,
    S: np.ndarray | float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    t: float,
    option_type: OptionType,
) -> np.ndarray:
    """
    Returns np.ndarray even if S is scalar.

    Black–Scholes European option price at time t (calendar time), maturity T.
    tau = T - t is time to maturity.
    """
    if T <= 0:
        raise ValueError(f"T must be > 0, got {T}")
    if t < 0:
        raise ValueError(f"t must be >= 0, got {t}")
    if t > T:
        raise ValueError(f"t must be <= T, got {t} > {T}")
    if K <= 0:
        raise ValueError(f"K must be > 0, got {K}")
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}")

    S = np.asarray(S, dtype=np.float64)
    tau = float(T - t)

    # tau==0 => payoff
    if tau == 0.0:
        if option_type == "call":
            return np.maximum(S - K, 0.0)
        else:
            return np.maximum(K - S, 0.0)

    out = np.empty_like(S, dtype=np.float64)

    # Handle S<=0 safely (edge case)
    mask0 = S <= 0
    mask = ~mask0

    if np.any(mask0):
        if option_type == "call":
            out[mask0] = 0.0
        else:
            out[mask0] = K * np.exp(-r * tau)

    if np.any(mask):
        Sm = S[mask]
        d1, d2 = bs_d1_d2(Sm, K, r, sigma, tau)
        Nd1 = _norm_cdf(d1)
        Nd2 = _norm_cdf(d2)

        if option_type == "call":
            out[mask] = Sm * Nd1 - K * np.exp(-r * tau) * Nd2
        else:
            out[mask] = K * np.exp(-r * tau) * _norm_cdf(-d2) - Sm * _norm_cdf(-d1)

    return out


def bs_european_greeks(
    *,
    S: np.ndarray | float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    t: float,
    option_type: OptionType,
) -> Dict[str, np.ndarray]:
    """
    Returns dict of arrays:
      price, delta=dV/dS, gamma=d2V/dS2, theta=dV/dt, vega=dV/dsigma, rho=dV/dr

    NOTE:
      tau = T - t (time to maturity)
      Standard “theta” in many finance texts is often reported as dV/dt (calendar time),
      which is negative of dV/dtau. We return dV/dt.
    """
    S = np.asarray(S, dtype=np.float64)
    tau = float(T - t)

    price = bs_european_price(S=S, K=K, r=r, sigma=sigma, T=T, t=t, option_type=option_type)

    # tau==0 => greeks unstable; return zeros (price is payoff already)
    if tau == 0.0:
        z = np.zeros_like(S, dtype=np.float64)
        return {"price": price, "delta": z, "gamma": z, "theta": z, "vega": z, "rho": z}

    # Safe init
    out_shape = S.shape
    delta = np.zeros(out_shape, dtype=np.float64)
    gamma = np.zeros(out_shape, dtype=np.float64)
    theta = np.zeros(out_shape, dtype=np.float64)
    vega = np.zeros(out_shape, dtype=np.float64)
    rho = np.zeros(out_shape, dtype=np.float64)

    mask = S > 0
    if not np.any(mask):
        return {"price": price, "delta": delta, "gamma": gamma, "theta": theta, "vega": vega, "rho": rho}

    Sm = S[mask]
    d1, d2 = bs_d1_d2(Sm, K, r, sigma, tau)
    pdf_d1 = _norm_pdf(d1)

    Nd1 = _norm_cdf(d1)
    Nd2 = _norm_cdf(d2)

    sqrt_tau = np.sqrt(tau)

    # Delta
    if option_type == "call":
        delta[mask] = Nd1
    else:
        delta[mask] = Nd1 - 1.0

    # Gamma (same call/put)
    gamma[mask] = pdf_d1 / (Sm * sigma * sqrt_tau)

    # Vega (same call/put)
    vega[mask] = Sm * pdf_d1 * sqrt_tau

    # Theta = dV/dt = - dV/dtau
    # For call: dC/dtau = + S φ(d1) σ/(2√tau) + rK e^{-r tau} N(d2)
    # => dC/dt  = -S φ(d1) σ/(2√tau) - rK e^{-r tau} N(d2)
    term1 = (Sm * pdf_d1 * sigma) / (2.0 * sqrt_tau)

    if option_type == "call":
        theta[mask] = -term1 - r * K * np.exp(-r * tau) * Nd2
        rho[mask] = K * tau * np.exp(-r * tau) * Nd2
    else:
        theta[mask] = -term1 + r * K * np.exp(-r * tau) * _norm_cdf(-d2)
        rho[mask] = -K * tau * np.exp(-r * tau) * _norm_cdf(-d2)

    return {"price": price, "delta": delta, "gamma": gamma, "theta": theta, "vega": vega, "rho": rho}
