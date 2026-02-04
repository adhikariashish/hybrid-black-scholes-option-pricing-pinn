# testing the gbm paths
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from src.data.gbm import GBMParams, simulate_gbm_paths

def test_gbm_shapes_and_positivity():
    params = GBMParams(s0=100.0, r=0.05, sigma=0.2, t=1.0, n_steps=252)
    paths = simulate_gbm_paths(params=params, n_paths=1000, seed=42)

    assert paths.shape == (1000, 253)
    assert np.allclose(paths[:, 0], 100.0)
    assert (paths > 0).all()

def test_gbm_terminal_mean_close_to_theory():
    s0, r, sigma, T, n_steps = 100.0, 0.05, 0.2, 1.0, 252
    params = GBMParams(s0=s0, r=r, sigma=sigma, t=T, n_steps=n_steps)

    paths = simulate_gbm_paths(params=params, n_paths=20000, seed=42)
    st = paths[:, -1]

    theory_mean = s0 * np.exp(r * T)
    mc_mean = st.mean()

    # Loose tolerance for MC noise
    assert abs(mc_mean - theory_mean) < 0.5

