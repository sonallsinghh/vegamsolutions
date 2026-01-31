# Run from project root: streamlit run app/ui.py
# UI only — no backend logic (no embeddings, no vector DB, no chunking).

import sys
import os
from pathlib import Path

# Ensure project root is on path (Streamlit may run with cwd != project root)
_root_from_file = Path(__file__).resolve().parent.parent
_cwd = os.getcwd()
for _root in (_root_from_file, _cwd):
    _root = str(_root)
    if _root not in sys.path:
        sys.path.insert(0, _root)

import streamlit as st
from app.ingest import load_files

st.title("Agentic RAG System")
st.text("System initialized")

uploaded_files = st.file_uploader(
    "Upload documents (.txt, .pdf, .xlsx, .xls)",
    type=["txt", "pdf", "xlsx", "xls"],
    accept_multiple_files=True,
)

documents = []
unsupported = []
if uploaded_files:
    documents, unsupported = load_files(uploaded_files)
    if unsupported:
        st.warning(f"Unsupported format (use .txt, .pdf, .xlsx, .xls): {', '.join(unsupported)}")
    if documents:
        st.write(f"Loaded {len(documents)} documents")

question = st.text_input("Ask a question")

st.button("Submit")
