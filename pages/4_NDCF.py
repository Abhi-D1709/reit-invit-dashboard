import streamlit as st
from utils.common import inject_global_css
from tabs.ndcf import render as render_ndcf

st.set_page_config(page_title="NDCF • REIT/InvIT", page_icon="📄", layout="wide")
inject_global_css()
render_ndcf()
