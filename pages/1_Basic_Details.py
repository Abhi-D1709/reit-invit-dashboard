# pages/5_Basic_Details.py
import streamlit as st
from utils.common import inject_global_css
from tabs.basic_details import render as render_directory

st.set_page_config(page_title="Basic Details", page_icon="📇", layout="wide")
inject_global_css()
render_directory()
