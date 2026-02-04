from pathlib import Path
import sys

# resolve project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config
from src.data.gbm import GBMParams
from src.data.make_mc_dataset import mc_price_european, save_mc_result_npz

def main() -> None:
    cfg = load_config()

    gbm_params = GBMParams(
        s0=cfg.data.s0,
        r=cfg.data.r,
        sigma=cfg.data.sigma,
        t=cfg.data.t,
        n_steps=cfg.data.n_steps,
    )

    option = cfg.data.option
    option_type = option["type"] if isinstance(option, dict) else option.type

    res = mc_price_european(
        gbm_params=gbm_params,
        k=cfg.data.k,
        option_type=option_type,
        n_paths=cfg.data.n_paths,
        seed=cfg.data.seed,
    )

    print("MC price:", res.price)

    save = getattr(cfg.data, "save_mc", False)
    if save:
        out_path = Path(cfg.data.dir.data, "mc_baseline.npz")
        meta = {
            "s0": cfg.data.s0,
            "k": cfg.data.k,
            "t": cfg.data.t,
            "r": cfg.data.r,
            "sigma": cfg.data.sigma,
            "n_paths": cfg.data.n_paths,
            "n_steps": cfg.data.n_steps,
            "seed": cfg.data.seed,
            "option_type": option_type,
        }
        save_mc_result_npz(result=res, out_path=out_path, meta=meta)
        print("Saved:", out_path)

if __name__ == "__main__":
    main()
