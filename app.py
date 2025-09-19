import streamlit as st
from utils.common import inject_global_css

st.set_page_config(page_title="REIT / InvIT Dashboard", page_icon="📊", layout="wide")
inject_global_css()

st.markdown("""
<div class="app-hero">
  <div class="big-title">REIT / InvIT Dashboard</div>
  <div class="subtle">Borrowings, Fund Raising, Trading, and NDCF.</div>
</div>
""", unsafe_allow_html=True)

st.write("Jump to a section:")
st.page_link("pages/1_Fund_Raising.py", label="💰 Fund Raising", icon="💰")
st.page_link("pages/2_Borrowings.py",   label="🏦 Borrowings",   icon="🏦")
st.page_link("pages/3_Trading.py",      label="📈 Trading",      icon="📈")
st.page_link("pages/4_NDCF.py",         label="📄 NDCF",         icon="📄")
