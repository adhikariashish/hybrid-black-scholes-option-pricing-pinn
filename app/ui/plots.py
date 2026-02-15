# app/ui/plots.py
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go


def overview_chart(*, s, bs, pinn, show_error: bool, show_bs: bool, spot: float | None) -> go.Figure:
    s = np.asarray(s)
    bs = np.asarray(bs)
    pinn = np.asarray(pinn)

    fig = go.Figure()
    if show_bs:
        fig.add_trace(go.Scatter(x=s, y=bs, mode="lines", name="Black-Scholes"))
    fig.add_trace(go.Scatter(x=s, y=pinn, mode="lines", name="PINN"))

    if spot is not None:
        fig.add_vline(x=float(spot), line_width=2, line_dash="dot")

    if show_error:
        err = pinn - bs
        fig.add_trace(go.Scatter(x=s, y=err, mode="lines", name="Error (PINN − BS)", yaxis="y2"))
        fig.update_layout(yaxis2=dict(title="Error", overlaying="y", side="right", showgrid=False))

    fig.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="S",
        yaxis_title="Price",
        title="PINN vs Black-Scholes",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def heatmap_figure(s, t, z, *, title: str, zlabel: str) -> go.Figure:
    fig = go.Figure(data=go.Heatmap(x=s, y=t, z=z, colorbar=dict(title=zlabel)))
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=35, b=10), xaxis_title="S", yaxis_title="t", title=title)
    return fig


def surface_figure(s, t, z, *, title: str) -> go.Figure:
    X, Y = np.meshgrid(s, t)
    fig = go.Figure(data=[go.Surface(x=X, y=Y, z=z)])
    fig.update_layout(
        height=450,
        margin=dict(l=10, r=10, t=35, b=10),
        scene=dict(xaxis_title="S", yaxis_title="t", zaxis_title="Price"),
        title=title,
    )
    return fig
