# app/ui/layout.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import streamlit as st


@dataclass(frozen=True)
class AppParams:
    # selection
    run_dir: str
    option_type: str  # "call" | "put"

    # trained contract (read-only)
    k: float
    r: float
    sigma: float
    t_maturity: float
    s0: float
    s_max_train: float

    # user-controlled (safe)
    s_mode: str             # "Single" | "Curve"
    s_value: float
    s_min: float
    s_max_plot: float
    s_steps: int

    show_bs: bool
    show_error: bool
    smooth_curves: bool

    # evaluation controls
    t_eval: float
    curve_resolution: int
    enable_csv_download: bool

def _discover_runs(default: str = "runs") -> list[str]:
    p = Path(default)
    if not p.exists() or not p.is_dir():
        return [default]
    dirs = sorted([str(d) for d in p.iterdir() if d.is_dir()])
    return dirs if dirs else [default]


def _load_config_snapshot(run_dir: Path, option_type: str) -> dict | None:
    """
    Loads runs/<run_name>/config_call.json or config_put.json
    """
    path = run_dir / f"config_{option_type}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _get_nested(d: dict, path: str, default=None):
    cur = d
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _to_float(x, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)

def render_sidebar(*, enable_time_slider: bool = True) -> AppParams:
    def _section(title: str) -> None:
        st.sidebar.markdown(
            f"""
    <div style="font-weight:600; margin-top:8px; margin-bottom:4px;">
    {title}
    </div>
    <div style="height:1px; background:rgba(255,255,255,0.15); margin-bottom:8px;"></div>
            """,
            unsafe_allow_html=True,
        )

    _section("Controls")

    run_choices = _discover_runs("runs")
    run_dir_str = st.sidebar.selectbox("Run directory", run_choices, index=0)
    run_dir = Path(run_dir_str)

    option_type = st.sidebar.radio("Option Type", ["call", "put"], horizontal=True)

    cfg = _load_config_snapshot(run_dir, option_type)

    cfg_opt_type = str(_get_nested(cfg or {}, "data.option.type", option_type)).lower().strip()

    k = _to_float(_get_nested(cfg or {}, "data.k", 100.0), 100.0)
    r = _to_float(_get_nested(cfg or {}, "data.r", 0.05), 0.05)
    sigma = _to_float(_get_nested(cfg or {}, "data.sigma", 0.20), 0.20)
    t_maturity = _to_float(_get_nested(cfg or {}, "data.t", 1.0), 1.0)
    s0 = _to_float(_get_nested(cfg or {}, "data.s0", 100.0), 100.0)
    s_max_train = _to_float(_get_nested(cfg or {}, "data.option.s_max", 300.0), 300.0)

    if cfg is None:
        st.sidebar.warning(
            f"Missing config_{option_type}.json in {run_dir_str}. "
            "Showing fallback values; contract display may be wrong."
        )

    if cfg is not None and cfg_opt_type in ("call", "put") and cfg_opt_type != option_type:
        st.sidebar.error(
            f"Config mismatch: config_{option_type}.json says option.type={cfg_opt_type!r}. "
            "Fix saved config or pick the matching option."
        )
    st.sidebar.divider()
    _section("Evaluation")

    # Time slider (optional)
    if enable_time_slider:
        # Evaluate at time t in [0, T]. (t=0 -> "today")
        t_eval = st.sidebar.slider(
            "Time (t)",
            min_value=0.0,
            max_value=float(t_maturity),
            value=0.0,
            step=float(t_maturity) / 100 if t_maturity > 0 else 0.01,
            help="Evaluation time. t=0 is today, t=T is maturity.",
        )
    else:
        t_eval = 0.0

    s_mode = st.sidebar.radio("S Mode", ["Single", "Curve"], horizontal=True)

    # Curve resolution control
    # (We keep your s_steps but expose a nicer control; maps to s_steps)
    resolution_map = {"Low (120)": 120, "Medium (200)": 200, "High (400)": 400}

    if s_mode == "Single":
        s_value = st.sidebar.number_input("Spot (S)", min_value=0.0, value=float(s0), step=1.0)
        s_min, s_max_plot, s_steps = float(max(0.0, s0 * 0.5)), float(s0 * 1.5), 200
        curve_resolution = 200
    else:
        s_min = st.sidebar.number_input("S min", min_value=0.0, value=float(max(0.0, s0 * 0.5)), step=1.0)
        s_max_plot = st.sidebar.number_input("S max", min_value=0.0, value=float(s0 * 1.5), step=1.0)

        res_choice = st.sidebar.selectbox("Curve resolution", list(resolution_map.keys()), index=1)
        curve_resolution = int(resolution_map[res_choice])

        # Keep the slider too? (optional) — I’d remove it to reduce clutter.
        s_steps = curve_resolution
        s_value = float((s_min + s_max_plot) / 2.0)

    st.sidebar.divider()
    _section("Display")

    show_bs = st.sidebar.toggle("Show Black-Scholes baseline", value=True)
    show_error = st.sidebar.toggle("Show error (PINN − BS)", value=True)
    smooth_curves = st.sidebar.toggle("Smooth curves", value=False)

    # Nice “future-proof” toggle (we’ll wire later)
    enable_csv_download = st.sidebar.toggle("Enable CSV download", value=False)

    return AppParams(
        run_dir=run_dir_str,
        option_type=option_type,
        k=k,
        r=r,
        sigma=sigma,
        t_maturity=t_maturity,
        s0=s0,
        s_max_train=s_max_train,
        s_mode=s_mode,
        s_value=float(s_value),
        s_min=float(s_min),
        s_max_plot=float(s_max_plot),
        s_steps=int(s_steps),
        show_bs=bool(show_bs),
        show_error=bool(show_error),
        smooth_curves=bool(smooth_curves),
        t_eval=float(t_eval),
        curve_resolution=int(curve_resolution),
        enable_csv_download=bool(enable_csv_download),
    )