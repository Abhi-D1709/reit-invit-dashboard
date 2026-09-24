# pages/12_Rules_Reference.py
import streamlit as st
from utils.common import inject_global_css
from tabs.rules_reference import render

st.set_page_config(page_title="Rules reference", page_icon="📜", layout="wide")
inject_global_css()
render()
