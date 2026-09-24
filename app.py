# app.py  -- entry point: page setup, theme styles, navigation
import streamlit as st

st.set_page_config(page_title="REIT / InvIT Dashboard", page_icon=":material/monitoring:", layout="wide")

from utils import navigation  # noqa: E402
from utils.chrome import render_data_banner  # noqa: E402
from utils.common import inject_global_css  # noqa: E402

inject_global_css()
page = st.navigation(navigation.build(), position="sidebar", expanded=True)
render_data_banner()
page.run()
