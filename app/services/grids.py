# app/services/grids.py
from __future__ import annotations

import numpy as np


def make_s_grid(params):
    if params.s_mode == "Single":
        smin = max(0.0, params.s_value * 0.5)
        smax = max(smin + 1.0, params.s_value * 1.5)
        return np.linspace(smin, smax, 200)
    return np.linspace(params.s_min, params.s_max_plot, params.s_steps)


def make_surface_grids(params):
    n_s = int(getattr(params, "curve_resolution", 200))
    n_t = 60
    s = np.linspace(max(0.0, params.s0 * 0.5), params.s0 * 1.5, n_s)
    t = np.linspace(0.0, params.t_maturity, n_t)
    return s, t
