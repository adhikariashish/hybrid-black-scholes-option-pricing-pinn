from __future__ import annotations

import argparse
from pathlib import Path

import sys

# resolve project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np


from src.utils.config import load_config
from src.models.pricing_pinn import PricingPINN
from src.physics.bs_closed_form import bs_european_price


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        type=str,
        default="runs/pinn_001/checkpoint_call.pt",
        help="Path to checkpoint file",
    )
    args = parser.parse_args()

    # ---- Load config ----
    cfg = load_config()

    torch.set_default_dtype(torch.float64)
    device = torch.device(getattr(cfg.train, "device", "cpu"))

    # ---- Load checkpoint ----
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)

    # ---- Rebuild model exactly as trained ----
    model = PricingPINN(**ckpt["model_cfg"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # ---- Problem parameters ----
    S0 = float(cfg.data.s0)
    K = float(cfg.data.k)
    r = float(cfg.data.r)
    sigma = float(cfg.data.sigma)
    T = float(cfg.data.t)
    option_type = cfg.data.option.type
    Smax = float(cfg.data.option.s_max)

    # ---- Normalized inputs (CRITICAL) ----
    s = torch.tensor([[S0 / Smax]], device=device)
    t = torch.tensor([[0.0 / T]], device=device)

    with torch.no_grad():
        v_pinn = model(torch.cat([s, t], dim=1)).item()

    # ---- Black-Scholes reference ----
    v_bs = bs_european_price(
        S=np.array([S0]),
        K=K,
        r=r,
        sigma=sigma,
        t=0.0,
        T=T,
        option_type=cfg.data.option.type,
    )[0]

    print("\n==== PRICE CHECK ====")
    print(f"PINN price  : {v_pinn:.6f}")
    print(f"BS price    : {v_bs:.6f}")
    print(f"Abs error   : {abs(v_pinn - v_bs):.6f}")
    print("=====================\n")


if __name__ == "__main__":
    main()
