import streamlit as st


st.set_page_config(
    page_title="Comparative Financial Analysis",
    page_icon="",
    layout="wide",
)

st.title("Comparative Financial Analysis Dashboard")
st.caption("NVIDIA, AMD, and Intel filing analysis with cited metrics and RAG Q&A.")

st.info(
    "Phase 1 scaffold is ready. Next phase: download SEC filings and populate "
    "`data/sources.csv`."
)

