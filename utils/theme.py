# utils/theme.py
"""Colours for the parts of the UI that Streamlit's theme does not reach (Plotly charts, styled tables).

Streamlit follows the visitor's light/dark setting; `is_dark()` reads which one is active so the
fixed colours below stay readable in both. Everything else should use the theme itself (no hex
colours in page code) or the translucent styles in `utils.common.inject_global_css`.
"""
from __future__ import annotations


def is_dark() -> bool:
    """True when the active Streamlit theme is dark (light when unknown, e.g. on the first run)."""
    try:
        import streamlit as st

        return getattr(st.context.theme, "type", None) == "dark"
    except Exception:
        return False


def text_color() -> str:
    return "#E6EDF5" if is_dark() else "#1A2433"


def grid_color() -> str:
    return "#2A3A4D" if is_dark() else "#EAF0F5"


def accent() -> str:
    """Primary series colour for charts (navy on light, a lighter blue on dark)."""
    return "#7FB2E5" if is_dark() else "#0F3D68"


def secondary() -> str:
    """Second series colour (gold), legible on both backgrounds."""
    return "#D2AE5C" if is_dark() else "#B08D3E"


def danger() -> str:
    return "#F08A8A" if is_dark() else "#A23B3B"


def table_row_styles() -> dict:
    """CSS for the styled ownership table rows, per row kind."""
    if is_dark():
        return {
            "grand_total": "background-color: #7FB2E5; color: #0E1621; font-weight: 600",
            "total": "background-color: #28405A; color: #E6EDF5; font-weight: 600",
            "subtotal": "background-color: #1E2B3A; color: #E6EDF5; font-weight: 600",
            "header": "font-weight: 700; color: #7FB2E5",
        }
    return {
        "grand_total": "background-color: #0F3D68; color: white; font-weight: 600",
        "total": "background-color: #D3E3EE; color: #1A2433; font-weight: 600",
        "subtotal": "background-color: #EAF0F5; color: #1A2433; font-weight: 600",
        "header": "font-weight: 700; color: #0F3D68",
    }
