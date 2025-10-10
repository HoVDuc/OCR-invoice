# app.py (Main page)

import streamlit as st
from pages import Upload, History, Statistics, Settings

st.set_page_config(
    page_title="Invoice Extractor",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="auto"
)

st.write("# Invoice Extractor!")
Upload.render()
