# Run from project root: streamlit run app/ui.py
# UI only — no backend logic connected.

import streamlit as st

st.title("Agentic RAG System")
st.text("System initialized")

uploaded_files = st.file_uploader(
    "Upload documents",
    type=None,
    accept_multiple_files=True,
)

question = st.text_input("Ask a question")

st.button("Submit")
