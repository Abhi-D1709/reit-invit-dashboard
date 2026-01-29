# pages/10_Investment.py
import streamlit as st
from tabs.investment import render

st.set_page_config(page_title="Investment Conditions", page_icon="💰", layout="wide")
render()