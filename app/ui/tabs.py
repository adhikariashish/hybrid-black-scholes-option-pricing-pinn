# app/ui/tabs.py
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.ui.components import section_title, info_banner, kpi_row
from app.ui.plots import overview_chart, heatmap_figure, surface_figure
from app.services.grids import make_s_grid, make_surface_grids
from app.services.pricing import bs_price_curve, pinn_price_curve
from app.services.loaders import load_history_safe


def render_tab_overview(params) -> None:

    s = make_s_grid(params)

    bs = bs_price_curve(params, s)
    pinn = pinn_price_curve(params, s)

    # KPI @ chosen spot
    s0 = params.s_value if params.s_mode == "Single" else float((params.s_min + params.s_max_plot) / 2.0)
    bs0 = float(bs_price_curve(params, [s0])[0])
    pinn0 = float(pinn_price_curve(params, [s0])[0])

    st.write("")
    kpi_row(bs0, pinn0)

    st.write("")
    fig = overview_chart(
        s=s,
        bs=bs,
        pinn=pinn,
        show_error=params.show_error,
        show_bs=params.show_bs,
        spot=s0 if params.s_mode == "Single" else None,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.write("")
    st.markdown("**Snapshot table**")
    df = pd.DataFrame({"S": s, "BS": bs, "PINN": pinn})
    df["Error"] = df["PINN"] - df["BS"]
    df["AbsError"] = df["Error"].abs()
    df["RelError(%)"] = df["AbsError"] / (df["BS"].abs() + 1e-12) * 100.0

    st.dataframe(df.head(300) if len(df) > 300 else df, use_container_width=True, height=260)


def render_tab_surface(params) -> None:
    s, t = make_surface_grids(params)
    bs_z, pinn_z = bs_price_curve(params, s, t_grid=t), pinn_price_curve(params, s, t_grid=t)

    mode = st.radio("View", ["Price (PINN)", "Price (BS)", "Error (PINN − BS)"], horizontal=True)
    use_3d = st.toggle("3D surface view", value=False)

    if mode == "Price (PINN)":
        z, title, zlabel = pinn_z, "PINN Price Surface", "Price"
    elif mode == "Price (BS)":
        z, title, zlabel = bs_z, "Black-Scholes Price Surface", "Price"
    else:
        z, title, zlabel = (pinn_z - bs_z), "Error Surface (PINN − BS)", "Error"

    fig = surface_figure(s, t, z, title=title) if use_3d else heatmap_figure(s, t, z, title=title, zlabel=zlabel)
    st.plotly_chart(fig, use_container_width=True)


def render_tab_diagnostics(params) -> None:
    hist = load_history_safe(params.run_dir, params.option_type)

    if hist is None:
        st.warning(f"No history found for {params.option_type.upper()} in {params.run_dir}.")
        return

    keys = [k for k in ["loss", "pde", "terminal", "bc"] if k in hist and isinstance(hist[k], list) and hist[k]]
    if not keys:
        st.warning("History loaded but has no usable series.")
        return

    # Total loss
    if "loss" in keys:
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=hist["loss"], mode="lines", name="loss"))
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=35, b=10), title="Training Loss", xaxis_title="Epoch", yaxis_title="Loss")
        st.plotly_chart(fig, use_container_width=True)

    # Components
    comp = [k for k in ["pde", "terminal", "bc"] if k in keys]
    if comp:
        fig = go.Figure()
        for k in comp:
            fig.add_trace(go.Scatter(y=hist[k], mode="lines", name=k))
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=35, b=10), title="Loss Components", xaxis_title="Epoch", yaxis_title="Component Loss")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**History (tail)**")
    st.dataframe(pd.DataFrame({k: hist[k] for k in keys}).tail(25), use_container_width=True, height=220)
