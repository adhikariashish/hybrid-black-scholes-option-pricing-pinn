from __future__ import annotations

from dataclasses import dataclass

import json
import time
from pathlib import Path

import numpy as np
import torch

# components
from src.utils.config import load_config
from src.models.pricing_pinn import PricingPINN
from src.losses.data_loss import mse_value_loss
from src.losses.physics_loss import physics_loss
from src.data.make_pinn_samples import make_pinn_dataset_from_cfg, sample_interior_points


#----------------------------
# helper function
#----------------------------

def _to_tensor(a: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(a).to(device=device, dtype=torch.get_default_dtype())


def main() -> None:
    #configuration for training
    cfg = load_config()

    # stability for 2nd derivative
    torch.set_default_dtype(torch.float64)

    device = torch.device(getattr(cfg.train, "device", "cpu"))

    run_dir = Path(getattr(cfg.train, "run_dir", "runs/pinn_v1"))
    run_dir.mkdir(parents=True, exist_ok=True)

    # model
    # build from model.yaml
    hidden_dim = int(cfg.model.hidden_dim)
    n_layers = int(cfg.model.n_hidden_layers)

    model_cfg = {
        "in_dimension": int(cfg.model.in_dimension),
        "hidden_sizes": tuple([hidden_dim] * n_layers),
        "activation": str(cfg.model.activation),
        "out_dimension": int(cfg.model.out_dimension),
    }

    model = PricingPINN(**model_cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg.train.lr))

    # PINN constraints
    T = float(cfg.data.t)
    s_max = float(getattr(cfg.data.option, "s_max", 300.0))

    pinn = make_pinn_dataset_from_cfg(cfg)

    # # interior collocation points (needs grad)
    # s_int = _to_tensor(pinn.interior.s/s_max, device)
    # t_int = _to_tensor(pinn.interior.t/T, device)
    # s_int.requires_grad_(True)
    # t_int.requires_grad_(True)


    #terminal
    s_term = _to_tensor(pinn.terminal.s/s_max, device)
    t_term = _to_tensor(pinn.terminal.t/T, device)
    v_term = _to_tensor(pinn.terminal.v, device)

    # boundary points
    b_low = pinn.boundary["low"]
    b_high = pinn.boundary["high"]

    s_low = _to_tensor(b_low.s/s_max, device)
    t_low = _to_tensor(b_low.t/T, device)
    v_low = _to_tensor(b_low.v, device)

    s_high = _to_tensor(b_high.s/s_max, device)
    t_high = _to_tensor(b_high.t/T, device)
    v_high = _to_tensor(b_high.v, device)

    # params
    r = float(cfg.data.r)
    sigma = float(cfg.data.sigma)

    # loss weights
    w_pde = float(cfg.train.weights.pde)
    w_term = float(cfg.train.weights.terminal)
    w_bc = float(cfg.train.weights.boundary)

    epochs = int(cfg.train.epochs)
    log_every = int(getattr(cfg.train, "log_every", 100))
    save_every = int(getattr(cfg.train, "save_every", 1000))

    history = {"loss": [], "pde": [], "terminal": [], "bc": []}

    t0 = time.time()
    resample_every = int(getattr(cfg.train.pinn, "resample_every", 100))

    for ep in range (1, epochs + 1):

        # resampling interior in every stated iteration
        if ep == 1 or (resample_every > 0 and ep % resample_every == 0):
            interior = sample_interior_points(
                n=cfg.train.pinn.n_interior,
                t_maturity=cfg.data.t,
                k=cfg.data.k,
                s_max=s_max,
                seed=None if cfg.data.seed is None else cfg.data.seed + 1000 + ep,
                sampler="mixture_k",
            )
            # interior collocation points (needs grad)
            s_int = _to_tensor(interior.s / s_max, device)
            t_int = _to_tensor(interior.t / T, device)
            s_int.requires_grad_(True)
            t_int.requires_grad_(True)


        opt.zero_grad(set_to_none=True)

        #physics loss
        lpde = physics_loss(model=model, s= s_int, t = t_int, r = r, sigma=sigma, T=T)

        # data loss
        lterm = mse_value_loss(model=model, s=s_term, t=t_term, v_true=v_term)

        #boundary term
        #low
        lbc_low = mse_value_loss(model=model, s=s_low, t=t_low, v_true=v_low)
        #high
        lbc_high = mse_value_loss(model=model, s=s_high, t=t_high, v_true=v_high)
        lbc = 0.5 * (lbc_low + lbc_high)

        #total loss
        loss = w_pde * lpde + w_term * lterm + w_bc * lbc
        loss.backward()
        opt.step()

        history["loss"].append(float(loss.detach().cpu()))
        history["pde"].append(float(lpde.detach().cpu()))
        history["terminal"].append(float(lterm.detach().cpu()))
        history["bc"].append(float(lbc.detach().cpu()))

        if ep == 1 or ep % log_every == 0:
            dt = time.time() - t0
            print(
                f"ep {ep:5d}/{epochs} | loss={history['loss'][-1]:.3e} "
                f"(pde={history['pde'][-1]:.2e}, term={history['terminal'][-1]:.2e}, bc={history['bc'][-1]:.2e}) "
                f"| {dt:.1f}s"
            )

        if ep % save_every == 0 or ep == epochs:
            torch.save(
                {
                    "epoch": ep,
                    "model_state": model.state_dict(),
                    "model_cfg": model_cfg,
                    "history": history,
                },
                run_dir / "checkpoint.pt",
            )
            with open(run_dir / "history.json", "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)

        # quick readout: V(S0, t=0)
    s0 = torch.tensor([[float(cfg.data.s0)]], device=device, dtype=torch.get_default_dtype())
    t00 = torch.tensor([[0.0]], device=device, dtype=torch.get_default_dtype())
    
    s0n = s0 / s_max
    t0n = t00 / T
    
    v0 = model(torch.cat([s0n, t0n], dim=1)).item()
    print(f"PINN price V(S0,t=0): {v0:.6f}")
    print(f"Saved run to: {run_dir}")


if __name__ == "__main__":
    main()


