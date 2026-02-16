# scripts/compare_pinn_bs_mc.py
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch

import sys

# resolve project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config
from src.models.pricing_pinn import PricingPINN
from src.physics.bs_closed_form import bs_european_price

# Use YOUR MC pipeline (the one that generated mc_baseline.npz)
# Adjust this import path to match your repo file name/location.
# Example if your file is src/data/mc_pricer.py:
#   from src.data.mc_pricer import mc_price_european, save_mc_result_npz
# Example if it's src/data/make_bc_dataset.py:
from src.data.make_mc_dataset import mc_price_european, save_mc_result_npz
from src.data.gbm import GBMParams

# ----------------------------
# Helpers
# ----------------------------
REQUIRED_META_KEYS = ("s0", "k", "t", "r", "sigma", "n_paths", "n_steps", "seed", "option_type")


def _as_float(x: Any) -> float:
    return float(x)


def _as_int(x: Any) -> int:
    return int(x)


def _meta_from_cfg(cfg) -> Dict[str, Any]:
    option_type = (
        cfg.data.option.type if hasattr(cfg.data, "option") else getattr(cfg.data, "option_type", "call")
    )
    return {
        "s0": _as_float(cfg.data.s0),
        "k": _as_float(cfg.data.k),
        "t": _as_float(cfg.data.t),
        "r": _as_float(cfg.data.r),
        "sigma": _as_float(cfg.data.sigma),
        "n_paths": _as_int(cfg.data.n_paths),
        "n_steps": _as_int(cfg.data.n_steps),
        "seed": _as_int(cfg.data.seed),
        "option_type": str(option_type).strip().lower(),
    }


def _load_npz_meta(npz_path: Path) -> Dict[str, Any]:
    data = np.load(npz_path, allow_pickle=True)
    meta = {}
    if "meta" in data:
        m = data["meta"].item()
        if isinstance(m, dict):
            meta = dict(m)
    return meta


def _std_error_ci95(discounted_payoff: np.ndarray) -> tuple[float, float]:
    x = np.asarray(discounted_payoff, dtype=float).reshape(-1)
    n = x.size
    if n <= 1:
        return float("nan"), float("nan")
    se = float(x.std(ddof=1) / np.sqrt(n))
    ci95 = 1.96 * se
    return se, ci95


def assert_meta_matches_cfg(
    *,
    cfg_meta: Dict[str, Any],
    npz_meta: Dict[str, Any],
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> None:
    """
    Raises ValueError if the saved MC baseline metadata doesn't match current config.
    """
    missing = [k for k in REQUIRED_META_KEYS if k not in npz_meta]
    if missing:
        raise ValueError(
            f"MC baseline meta missing keys: {missing}. "
            f"Re-generate the baseline with this script or ensure save_mc_result_npz stores meta."
        )

    # float keys: tolerate tiny fp drift
    float_keys = ("s0", "k", "t", "r", "sigma")
    for k in float_keys:
        a = float(cfg_meta[k])
        b = float(npz_meta[k])
        if not np.isclose(a, b, rtol=rtol, atol=atol):
            raise ValueError(f"MC baseline meta mismatch for {k}: cfg={a} vs npz={b}")

    # int keys: must match exactly
    int_keys = ("n_paths", "n_steps", "seed")
    for k in int_keys:
        a = int(cfg_meta[k])
        b = int(npz_meta[k])
        if a != b:
            raise ValueError(f"MC baseline meta mismatch for {k}: cfg={a} vs npz={b}")

    # option type: must match exactly (normalized)
    a = str(cfg_meta["option_type"]).strip().lower()
    b = str(npz_meta["option_type"]).strip().lower()
    if a != b:
        raise ValueError(f"MC baseline meta mismatch for option_type: cfg={a} vs npz={b}")


def ensure_mc_baseline(
    *,
    cfg,
    out_path: Path,
    force: bool = False,
) -> Dict[str, Any]:
    """
    Ensures data/processed/mc_baseline.npz exists and matches current cfg.
    If missing or mismatch (or force=True), regenerate it using current cfg.
    Returns a dict with keys: price, se, ci95, meta
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_meta = _meta_from_cfg(cfg)

    def _regen() -> Dict[str, Any]:
        gbm = GBMParams(
            s0=float(cfg.data.s0),
            r=float(cfg.data.r),
            sigma=float(cfg.data.sigma),
            t=float(cfg.data.t),
            n_steps=int(cfg.data.n_steps),
        )
        res = mc_price_european(
            gbm_params=gbm,
            k=float(cfg.data.k),
            option_type=cfg_meta["option_type"],
            n_paths=int(cfg.data.n_paths),
            seed=int(cfg.data.seed),
            r=float(cfg.data.r),
        )
        meta = dict(cfg_meta)
        save_mc_result_npz(result=res, out_path=out_path, meta=meta)
        se, ci95 = _std_error_ci95(res.discounted_payoff)
        return {"price": float(res.price), "se": se, "ci95": ci95, "meta": meta, "regen": True}

    if force or (not out_path.exists()):
        return _regen()

    # validate existing
    data = np.load(out_path, allow_pickle=True)
    npz_meta = _load_npz_meta(out_path)
    try:
        assert_meta_matches_cfg(cfg_meta=cfg_meta, npz_meta=npz_meta)
    except ValueError:
        # mismatch -> regenerate to avoid invalid comparisons
        return _regen()

    price = float(data["price"])
    discounted = data["discounted_payoff"].astype(float)
    se, ci95 = _std_error_ci95(discounted)
    return {"price": price, "se": se, "ci95": ci95, "meta": npz_meta, "regen": False}


def pinn_prices_at(
    *,
    model: torch.nn.Module,
    s_vals: np.ndarray,
    t: float,
    device: torch.device,
    S_max: float,
    T: float,
) -> np.ndarray:
    """
    Assumes your model expects normalized inputs: (S/S_max, t/T).
    """
    s = torch.from_numpy(s_vals.reshape(-1, 1)).to(device=device, dtype=torch.get_default_dtype())
    tt = torch.full_like(s, float(t))
    x = torch.cat([s / float(S_max), tt / float(T)], dim=1)
    with torch.no_grad():
        v = model(x).detach().cpu().numpy().reshape(-1)
    return v


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="runs/pinn_001/checkpoint_call.pt")
    ap.add_argument("--mc_npz", type=str, default="data/processed/mc_baseline.npz")
    ap.add_argument("--out_dir", type=str, default="reports/compare")
    ap.add_argument("--n_grid", type=int, default=12)
    ap.add_argument("--t", type=float, default=0.0, help="time for curve table")
    ap.add_argument("--force_mc", action="store_true", help="force regenerate MC baseline from current config")
    args = ap.parse_args()

    cfg = load_config()

    torch.set_default_dtype(torch.float64)
    device = torch.device(getattr(cfg.train, "device", "cpu"))

    # ---- Load PINN checkpoint
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    if "model_cfg" not in ckpt:
        raise KeyError("Checkpoint missing 'model_cfg'. Save it during training to enable reproducible evaluation.")

    model = PricingPINN(**ckpt["model_cfg"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # ---- Params (must match across PINN/BS/MC)
    S0 = float(cfg.data.s0)
    K = float(cfg.data.k)
    r = float(cfg.data.r)
    sigma = float(cfg.data.sigma)
    T = float(cfg.data.t)
    option_type = (
        cfg.data.option.type if hasattr(cfg.data, "option") else "call"
    )
    option_type = str(option_type).strip().lower()

    # Consistent with your evaluation domain (3K)
    S_MAX = 3.0 * K

    # ---- Ensure MC baseline exists and matches current cfg
    mc_info = ensure_mc_baseline(
        cfg=cfg,
        out_path=Path(args.mc_npz),
        force=bool(args.force_mc),
    )

    # ---- S0 point check at t=0
    pinn_s0 = float(pinn_prices_at(model=model, s_vals=np.array([S0], dtype=np.float64), t=0.0,
                                   device=device, S_max=S_MAX, T=T)[0])
    bs_s0 = float(
        bs_european_price(
            S=np.array([S0], dtype=np.float64),
            K=K,
            r=r,
            sigma=sigma,
            t=0.0,
            T=T,
            option_type=cfg.data.option.type,
        )[0]
    )

    # ---- Curve table around strike
    s_vals = np.linspace(0.6 * K, 1.4 * K, args.n_grid, dtype=np.float64)
    if np.all(np.abs(s_vals - S0) > 1e-9):
        s_vals = np.sort(np.unique(np.append(s_vals, S0)))

    t_curve = float(args.t)

    pinn_vals = pinn_prices_at(model=model, s_vals=s_vals, t=t_curve, device=device, S_max=S_MAX, T=T)
    bs_vals = bs_european_price(S=s_vals, K=K, r=r, sigma=sigma, t=t_curve, T=T, option_type=option_type)

    df = pd.DataFrame(
        {
            "t": t_curve,
            "S": s_vals,
            "PINN": pinn_vals,
            "BS": bs_vals,
            "abs_err_pinn_bs": np.abs(pinn_vals - bs_vals),
        }
    )

    rmse = float(np.sqrt(np.mean((pinn_vals - bs_vals) ** 2)))
    max_abs = float(np.max(np.abs(pinn_vals - bs_vals)))

    suffix = "call" if cfg.data.option.type == "call" else "put"

    # Output dir
    if args.out_dir is not None:
        out_dir = Path(args.out_dir) / suffix
    else:
        # default: reports/figures/pinn_001/call (or put)
        run_name = ckpt_path.parent.name
        out_dir = Path("reports/compare") / run_name / suffix

    out_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_dir / "compare_table.csv", index=False)

    summary = {
        "ckpt": str(ckpt_path),
        "mc_npz": str(Path(args.mc_npz)),
        "mc_regenerated": bool(mc_info.get("regen", False)),
        "params": {
            "S0": S0,
            "K": K,
            "r": r,
            "sigma": sigma,
            "T": T,
            "option_type": option_type,
            "S_MAX": S_MAX,
        },
        "S0_check": {
            "PINN": pinn_s0,
            "BS": bs_s0,
            "abs_err_pinn_bs": abs(pinn_s0 - bs_s0),
            "MC": mc_info["price"],
            "MC_se": mc_info["se"],
            "MC_ci95": mc_info["ci95"],
            "abs_err_mc_bs": abs(mc_info["price"] - bs_s0),
        },
        "curve_metrics": {
            "t": t_curve,
            "rmse_pinn_vs_bs": rmse,
            "max_abs_pinn_vs_bs": max_abs,
        },
        "files": {
            "table_csv": str((out_dir / "compare_table.csv").as_posix()),
            "summary_json": str((out_dir / "summary.json").as_posix()),
        },
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Nice console output
    print("==== PINN vs BS vs MC (European) ====")
    print(f"Checkpoint     : {ckpt_path}")
    print(f"MC baseline    : {Path(args.mc_npz)} (regenerated={summary['mc_regenerated']})")
    print(f"Params         : S0={S0}, K={K}, r={r}, sigma={sigma}, T={T}, opt={option_type}")
    print("")
    print("---- S0 (t=0) ----")
    print(f"PINN           : {pinn_s0:.6f}")
    print(f"BS             : {bs_s0:.6f}")
    print(f"|PINN-BS|      : {abs(pinn_s0 - bs_s0):.6f}")
    print(f"MC             : {mc_info['price']:.6f} ± {mc_info['ci95']:.6f} (95% CI)")
    print(f"|MC-BS|        : {abs(mc_info['price'] - bs_s0):.6f}")
    print("")
    print(f"---- Curve (t={t_curve}) ----")
    print(f"RMSE(PINN vs BS): {rmse:.6f}")
    print(f"MaxAbs         : {max_abs:.6f}")
    print("")
    print(f"Saved table    : {out_dir / 'compare_table.csv'}")
    print(f"Saved summary  : {out_dir / 'summary.json'}")
    print("====================================")


if __name__ == "__main__":
    main()
