from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.utils.config import load_config
from src.models.pricing_pinn import PricingPINN
from src.physics.bs_closed_form import OptionType, bs_european_price, bs_european_greeks


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def compute_pinn_price_and_greeks(
    model: torch.nn.Module,
    s_phys: torch.Tensor,   # (n,1) in real S
    t_phys: torch.Tensor,   # (n,1) in real t
    *,
    s_max: float,
    T: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Model was trained on normalized inputs:
        u = S / S_max
        v = t / T

    Returns in REAL UNITS (after chain rule):
        V, Delta=dV/dS, Gamma=d2V/dS2, Theta=dV/dt
    """
    # normalize inputs for the model
    u = (s_phys / float(s_max)).clone().detach().requires_grad_(True)
    v = (t_phys / float(T)).clone().detach().requires_grad_(True)

    x = torch.cat([u, v], dim=1)  # (n,2)
    V = model(x)                 # (n,1)

    ones = torch.ones_like(V)

    dV_du = torch.autograd.grad(V, u, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    dV_dv = torch.autograd.grad(V, v, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    d2V_du2 = torch.autograd.grad(
        dV_du, u, grad_outputs=torch.ones_like(dV_du), create_graph=True, retain_graph=True
    )[0]

    # Chain rule back to physical units:
    # dV/dS = dV/du * du/dS = dV/du * (1/S_max)
    # d2V/dS2 = d2V/du2 * (1/S_max^2)
    # dV/dt = dV/dv * (1/T)
    Delta = dV_du / float(s_max)
    Gamma = d2V_du2 / (float(s_max) ** 2)
    Theta = dV_dv / float(T)

    return V, Delta, Gamma, Theta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="runs/pinn_v4/checkpoint.pt")
    parser.add_argument("--n", type=int, default=400, help="number of S points")
    parser.add_argument("--s_min", type=float, default=0.0)
    parser.add_argument("--s_max", type=float, default=None, help="defaults to 3*K if not provided")
    parser.add_argument("--times", type=str, default="0.0,0.5", help="comma list as fractions of T")
    parser.add_argument("--out_dir", type=str, default=None, help="override output fig dir")
    args = parser.parse_args()

    cfg = load_config()

    torch.set_default_dtype(torch.float64)
    device = torch.device(getattr(cfg.train, "device", "cpu"))

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)

    if "model_cfg" not in ckpt:
        raise KeyError(
            "Checkpoint missing 'model_cfg'. Save it during training so evaluation can rebuild the same model."
        )

    model = PricingPINN(**ckpt["model_cfg"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Params
    K = float(cfg.data.k)
    T = float(cfg.data.t)
    r = float(cfg.data.r)
    sigma = float(cfg.data.sigma)

    option_cfg = cfg.data.option
    option_type = cast(OptionType, option_cfg.type)  # "call" or "put"

    # Important: S_max must match training normalization
    S_MAX = float(args.s_max) if args.s_max is not None else 3.0 * K

    # S grid in PHYSICAL units (this is what BS uses)
    s_grid = np.linspace(args.s_min, S_MAX, args.n, dtype=np.float64).reshape(-1, 1)

    fracs = [float(x.strip()) for x in args.times.split(",") if x.strip()]
    t_list = [f * T for f in fracs]

    # Output dir
    if args.out_dir is not None:
        fig_dir = Path(args.out_dir)
    else:
        # default like reports/figures/pinn_v4 (use run folder name if you want)
        fig_dir = Path("reports/figures") / ckpt_path.parent.name
    fig_dir.mkdir(parents=True, exist_ok=True)

    for tt in t_list:
        # torch inputs in PHYSICAL units
        s_t = torch.from_numpy(s_grid).to(device)
        t_t = torch.full_like(s_t, fill_value=float(tt)).to(device)

        # PINN outputs (real units)
        V, Delta, Gamma, Theta = compute_pinn_price_and_greeks(
            model, s_t, t_t, s_max=S_MAX, T=T
        )

        V_np = _to_np(V).reshape(-1)
        d_np = _to_np(Delta).reshape(-1)
        g_np = _to_np(Gamma).reshape(-1)
        th_np = _to_np(Theta).reshape(-1)

        # BS reference (PHYSICAL units)
        V_bs = bs_european_price(S=s_grid.reshape(-1), K=K, r=r, sigma=sigma, T=T, t=float(tt), option_type=option_type)
        G_bs = bs_european_greeks(S=s_grid.reshape(-1), K=K, r=r, sigma=sigma, T=T, t=float(tt), option_type=option_type)

        d_bs = G_bs["delta"].reshape(-1)
        g_bs = G_bs["gamma"].reshape(-1)
        th_bs = G_bs["theta"].reshape(-1)  # dV/dt (matches our convention)

        # Errors
        err_V = V_np - V_bs
        err_d = d_np - d_bs
        err_g = g_np - g_bs
        err_th = th_np - th_bs

        # Metrics
        rmse_V = _rmse(V_np, V_bs)
        rmse_d = _rmse(d_np, d_bs)
        rmse_g = _rmse(g_np, g_bs)
        rmse_th = _rmse(th_np, th_bs)
        max_abs_V = float(np.max(np.abs(err_V)))

        worst_idx = int(np.argmax(np.abs(err_V)))
        worst_S = float(s_grid.reshape(-1)[worst_idx])
        print(
            f"[t={tt:.3f}] price RMSE={rmse_V:.6f}, max_abs_err={max_abs_V:.6f} | "
            f"delta RMSE={rmse_d:.6f} | gamma RMSE={rmse_g:.6f} | theta RMSE={rmse_th:.6f}"
        )
        print(
            f"[t={tt:.3f}] worst at S={worst_S:.6f}: PINN={V_np[worst_idx]:.6f}, BS={V_bs[worst_idx]:.6f}, err={err_V[worst_idx]:.6f}"
        )

        suffix = f"{tt:.3f}"

        # --- Plots ---
        Sx = s_grid.reshape(-1)

        # Price
        plt.figure()
        plt.plot(Sx, V_np, label="PINN")
        plt.plot(Sx, V_bs, label="BS")
        plt.title(f"Price vs S | t={tt:.3f} (T={T})")
        plt.xlabel("S")
        plt.ylabel("V")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / f"price_{suffix}.png", dpi=150)
        plt.close()

        # Price error
        plt.figure()
        plt.plot(Sx, err_V)
        plt.title(f"Price Error (PINN - BS) | t={tt:.3f} (T={T})")
        plt.xlabel("S")
        plt.ylabel("Error")
        plt.tight_layout()
        plt.savefig(fig_dir / f"price_err_{suffix}.png", dpi=150)
        plt.close()

        # Delta (compare)
        plt.figure()
        plt.plot(Sx, d_np, label="PINN")
        plt.plot(Sx, d_bs, label="BS")
        plt.title(f"Delta dV/dS | t={tt:.3f} (T={T})")
        plt.xlabel("S")
        plt.ylabel("Delta")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / f"delta_{suffix}.png", dpi=150)
        plt.close()

        # Gamma (compare)
        plt.figure()
        plt.plot(Sx, g_np, label="PINN")
        plt.plot(Sx, g_bs, label="BS")
        plt.title(f"Gamma d2V/dS2 | t={tt:.3f} (T={T})")
        plt.xlabel("S")
        plt.ylabel("Gamma")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / f"gamma_{suffix}.png", dpi=150)
        plt.close()

        # Theta (compare)
        plt.figure()
        plt.plot(Sx, th_np, label="PINN")
        plt.plot(Sx, th_bs, label="BS")
        plt.title(f"Theta dV/dt | t={tt:.3f} (T={T})")
        plt.xlabel("S")
        plt.ylabel("dV/dt")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / f"theta_{suffix}.png", dpi=150)
        plt.close()

    print(f"Saved figures to: {fig_dir}")


if __name__ == "__main__":
    main()
