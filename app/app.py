from __future__ import annotations

import streamlit as st
import sys
from pathlib import Path

# resolve project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from app.ui.theme import inject_theme_css
from app.ui.layout import render_sidebar
from app.ui.components import contract_line
from app.ui.tabs import render_tab_overview, render_tab_surface, render_tab_diagnostics


def main() -> None:
    st.set_page_config(page_title="Hybrid Black-Scholes PINN", page_icon="📈", layout="wide")
    inject_theme_css()

    params = render_sidebar()

    st.title("Hybrid Black-Scholes Option Pricing (PINN + Closed-Form)")
    st.caption(f"Run: {params.run_dir} • Option: {params.option_type.upper()}")

    st.write("")
    contract_line(params)

    tab1, tab2, tab3 = st.tabs(["Overview", "Surface / Heatmap", "Diagnostics"])

    with tab1:
        render_tab_overview(params)

    with tab2:
        render_tab_surface(params)

    with tab3:
        render_tab_diagnostics(params)


if __name__ == "__main__":
    main()
