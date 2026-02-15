# app/ui/components.py
from __future__ import annotations

import streamlit as st


def section_title(text: str) -> None:
    st.markdown(f'<div class="bsm-section">{text}</div>', unsafe_allow_html=True)


def info_banner(text: str) -> None:
    st.markdown(f'<div class="bsm-banner">{text}</div>', unsafe_allow_html=True)


def kpi_card(title: str, value: str, *, delta: str | None = None, delta_sign: str = "neu") -> None:
    """
    delta_sign: "pos" | "neg" | "neu"
    """
    delta_html = ""
    if delta is not None:
        delta_html = f'<div class="bsm-delta {delta_sign}">{delta}</div>'

    st.markdown(
        f"""
<div class="bsm-card">
  <div class="bsm-card-title">{title}</div>
  <div class="bsm-card-value">{value}</div>
  {delta_html}
</div>
        """,
        unsafe_allow_html=True,
    )
def contract_panel(
    *,
    option_type: str,
    k: float,
    r: float,
    sigma: float,
    t_maturity: float,
    s_max: float,
) -> None:
    st.markdown(
        f"""
<div class="bsm-card" style="padding:10px 14px; ">
  <div style="font-size:0.9rem; line-height:1.4; ">
    <b>Contract:</b>
    Option : <b>{option_type.upper()}</b> &nbsp;|&nbsp;
    K : {k:g} &nbsp;|&nbsp;
    r : {r:g} &nbsp;|&nbsp;
    σ : {sigma:g} &nbsp;|&nbsp;
    T : {t_maturity:g} &nbsp;|&nbsp;
    S<sub>max</sub>: {s_max:g}
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )
def kpi_row(bs0: float, pinn0: float) -> None:
    abs_err = abs(pinn0 - bs0)
    rel_err = abs_err / (abs(bs0) + 1e-12)

    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1:
        kpi_card("BS Price", f"${bs0:,.4f}", delta=None, delta_sign="neu")
    with c2:
        sign = "pos" if pinn0 >= bs0 else "neg"
        kpi_card("PINN Price", f"${pinn0:,.4f}", delta=f"{(pinn0 - bs0):+.4f}", delta_sign=sign)
    with c3:
        kpi_card("Abs Error", f"{abs_err:.6f}", delta="↓ better", delta_sign="pos")
    with c4:
        kpi_card("Rel Error", f"{(rel_err * 100):.3f}%", delta="target < 1%", delta_sign="neu")

def contract_line(params) -> None:
    st.markdown(
        f"""
<div class="bsm-card" style="padding:10px 14px;">
  <div style="font-size:0.92rem; line-height:1.4;">
    <b>Contract:</b>
    Option : <b>{params.option_type.upper()}</b> &nbsp;|&nbsp;
    K : {params.k:g} &nbsp;|&nbsp;
    r : {params.r:g} &nbsp;|&nbsp;
    σ : {params.sigma:g} &nbsp;|&nbsp;
    T : {params.t_maturity:g} &nbsp;|&nbsp;
    S<sub>max</sub> : {params.s_max_train:g}
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )
