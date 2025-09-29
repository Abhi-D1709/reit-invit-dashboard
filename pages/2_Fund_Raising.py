import streamlit as st
from utils.common import inject_global_css
from tabs.fundraising import render as render_fund

st.set_page_config(page_title="Fund Raising • REIT/InvIT", page_icon="💰", layout="wide")
inject_global_css()
render_fund()
