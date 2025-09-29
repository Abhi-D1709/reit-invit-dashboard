# pages/6_Sponsor_Holding.py
import streamlit as st
from tabs.sponsor_holding import render as render_sponsor

st.set_page_config(page_title="Sponsor Holding", page_icon="🧩", layout="wide")
render_sponsor()
