# pages/7_Valuation.py
import streamlit as st
from tabs.valuation import render as render_valuation

st.set_page_config(page_title="Valuation", page_icon= "📐", layout="wide")
render_valuation()