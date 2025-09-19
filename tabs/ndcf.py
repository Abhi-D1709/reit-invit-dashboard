# tabs/ndcf.py
import streamlit as st

def render():
    st.header("NDCF")
    segment = st.selectbox("Select Segment", ["REIT", "InvIT"], key="seg_ndcf")
    st.info(f"{segment} NDCF dashboard will appear here once data is available.")
