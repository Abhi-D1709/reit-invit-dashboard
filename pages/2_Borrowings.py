import streamlit as st
from utils.common import inject_global_css
from tabs.borrowings import render as render_borrow

st.set_page_config(page_title="Borrowings • REIT/InvIT", page_icon="🏦", layout="wide")
inject_global_css()
render_borrow()
