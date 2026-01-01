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
st.page_link("pages/1_Basic_Details.py", label="Basic Details", icon="📇")
st.page_link("pages/2_Fund_Raising.py", label="Fund Raising", icon="💰")
st.page_link("pages/3_Borrowings.py",   label="Borrowings",   icon="🏦")
st.page_link("pages/4_Trading.py",      label="Trading",      icon="📈")
st.page_link("pages/5_NDCF.py",         label="NDCF",         icon="📄")
st.page_link("pages/6_Sponsor_Holding.py",         label="Sponsor Holding",         icon="🧩")
st.page_link("pages/7_Governance.py",         label="Governance",         icon="🧭")
st.page_link("pages/8_Valuation.py",         label="Valuation",         icon="📐")