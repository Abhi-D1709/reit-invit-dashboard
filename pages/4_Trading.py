import streamlit as st
from utils.common import inject_global_css
from tabs.trading import render as render_trade

st.set_page_config(page_title="Trading • REIT/InvIT", page_icon="📈", layout="wide")
inject_global_css()
render_trade()
