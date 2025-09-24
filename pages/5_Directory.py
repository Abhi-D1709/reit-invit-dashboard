# pages/5_Directory.py
import streamlit as st
from utils.common import inject_global_css
from tabs.directory import render as render_directory

st.set_page_config(page_title="Basic Details", page_icon="📇", layout="wide")
inject_global_css()
render_directory()
