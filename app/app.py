import json
from pathlib import Path

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="PINN vs BS vs MC", layout="wide")

st.title("Hybrid Black–Scholes Option Pricing (PINN)")

compare_dir = Path("reports/compare")
summary_path = compare_dir / "summary.json"
csv_path = compare_dir / "compare_prices.csv"

if not summary_path.exists() or not csv_path.exists():
    st.error("Missing reports/compare outputs. Run: python scripts/compare_mc_bs_pinn.py")
    st.stop()

summary = json.loads(summary_path.read_text(encoding="utf-8"))
df = pd.read_csv(csv_path)

# --- KPI row ---
s0 = summary["S0_prices"]
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("PINN @ S0", f"{s0['pinn']:.4f}")
c2.metric("BS @ S0", f"{s0['bs']:.4f}")
c3.metric("MC @ S0", f"{s0['mc']:.4f}", help=f"±{s0['mc_ci95']:.4f} (95% CI)")
c4.metric("|PINN−BS|", f"{s0['abs_err_pinn_bs']:.4f}")
c5.metric("|MC−BS|", f"{s0['abs_err_mc_bs']:.4f}")

st.divider()

# --- Controls ---
times = sorted(df["t"].unique().tolist())
t_sel = st.selectbox("Select time t", times, index=0)

dft = df[df["t"] == t_sel].sort_values("S")

# --- Plots ---
colA, colB = st.columns([2, 1])

with colA:
    st.subheader("Price curve")
    fig = plt.figure()
    plt.plot(dft["S"], dft["PINN"], label="PINN")
    plt.plot(dft["S"], dft["BS"], label="BS")
    plt.xlabel("S")
    plt.ylabel("V")
    plt.legend()
    st.pyplot(fig, clear_figure=True)

    st.subheader("Absolute error |PINN−BS|")
    fig2 = plt.figure()
    plt.plot(dft["S"], dft["abs_err_pinn_bs"])
    plt.xlabel("S")
    plt.ylabel("Abs Error")
    st.pyplot(fig2, clear_figure=True)

with colB:
    st.subheader("Comparison table")
    show = dft.copy()
    show["PINN"] = show["PINN"].map(lambda x: f"{x:.6f}")
    show["BS"] = show["BS"].map(lambda x: f"{x:.6f}")
    show["abs_err_pinn_bs"] = show["abs_err_pinn_bs"].map(lambda x: f"{x:.6f}")
    st.dataframe(show[["S", "PINN", "BS", "abs_err_pinn_bs"]], use_container_width=True)

st.caption("MC is reported at S0 with 95% CI; curves are PINN vs BS across S.")
