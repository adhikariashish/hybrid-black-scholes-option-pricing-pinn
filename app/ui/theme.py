# app/ui/theme.py
from __future__ import annotations

import streamlit as st


def inject_theme_css() -> None:
    """
    Inject CSS for a fixed dashboard feel:
    - tighter top padding
    - nicer cards
    - delta colors
    - optional "app frame" height control
    """
    st.markdown(
        """
<style>
/* --- Base layout tightening --- */
div.block-container {
    padding-top: 0.8rem;
    padding-bottom: 1rem;
}

/* Remove extra whitespace around main sections */
section.main > div {
    padding-top: 0.5rem;
}

/* Sidebar spacing */
section[data-testid="stSidebar"] div.block-container > div {
    margin-bottom: 0.4rem;
}

section[data-testid="stSidebar"] p {
    margin-bottom: 0.25rem;
}
section[data-testid="stSidebar"] hr {
    margin-top: 0.3rem;
    margin-bottom: 0.5rem;
}
/* --- Card system --- */
.bsm-card {
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 14px 14px;
    background: rgba(255,255,255,0.03);
    box-shadow: 0 8px 22px rgba(0,0,0,0.12);
}

.bsm-card-title {
    font-size: 0.86rem;
    opacity: 0.85;
    margin: 0 0 4px 0;
}

.bsm-card-value {
    font-size: 1.35rem;
    font-weight: 700;
    margin: 0;
    line-height: 1.2;
}

.bsm-delta {
    font-size: 0.86rem;
    margin-top: 6px;
    opacity: 0.9;
}

.bsm-delta.pos { color: #2ecc71; }   /* green */
.bsm-delta.neg { color: #e74c3c; }   /* red */
.bsm-delta.neu { color: #95a5a6; }   /* gray */

/* --- Section header --- */
.bsm-section {
    font-size: 1.05rem;
    font-weight: 700;
    margin: 0.25rem 0 0.75rem 0;
}

/* --- Info banner --- */
.bsm-banner {
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 14px;
    padding: 10px 12px;
    background: rgba(52, 152, 219, 0.12);
}

/* --- Tabs spacing --- */
div[data-baseweb="tab-list"] button {
    padding-top: 10px;
    padding-bottom: 10px;
}

/* Optional: Reduce header space on top of tabs */
div[data-testid="stTabs"] {
    margin-top: 0.25rem;
}


/* --- “Web app” feel: keep the main content compact --- */
/* Note: Streamlit always scrolls the page if content overflows.
   We'll keep content short per tab and make charts use container width. */
</style>
        """,
        unsafe_allow_html=True,
    )
