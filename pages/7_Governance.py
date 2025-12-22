# pages/7_Governance.py
import streamlit as st
from utils.common import inject_global_css
from tabs.governance import render as render_governance

st.set_page_config(page_title="Governance • REIT/InvIT", page_icon="🧭", layout="wide")
inject_global_css()
render_governance()
