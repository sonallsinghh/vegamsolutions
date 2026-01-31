# Run from project root: streamlit run app/ui.py
# UI talks to backend API (POST /upload, WebSocket /ws/query). Frontend client only.

import json
import os
import sys
from pathlib import Path

# Ensure project root is on path (Streamlit may run with cwd != project root)
_root_from_file = Path(__file__).resolve().parent.parent
_cwd = os.getcwd()
for _root in (_root_from_file, _cwd):
    _root = str(_root)
    if _root not in sys.path:
        sys.path.insert(0, _root)

import streamlit as st
import requests
import websocket

# Backend config
API_BASE = os.environ.get("API_BASE", "http://localhost:8000")
WS_URL = os.environ.get("WS_URL", "ws://localhost:8000/ws/query")

st.title("Agentic RAG System")
st.text("System initialized")

uploaded_files = st.file_uploader(
    "Upload documents (.txt, .pdf, .xlsx, .xls)",
    type=["txt", "pdf", "xlsx", "xls"],
    accept_multiple_files=True,
)

# Upload flow: POST to backend instead of local load_files
if uploaded_files:
    try:
        files = []
        for uf in uploaded_files:
            uf.seek(0)
            files.append(("files", (uf.name, uf.read())))
        r = requests.post(f"{API_BASE}/upload", files=files, timeout=30)
        if r.ok:
            data = r.json()
            st.success(f"Files uploaded successfully ({data.get('files_saved', 0)} saved)")
        else:
            st.error(f"Upload failed: {r.status_code} — {r.text[:200]}")
    except requests.RequestException as e:
        st.error(f"Upload failed: {e}")

question = st.text_input("Ask a question")

status_box = st.empty()
answer_box = st.empty()

if st.button("Submit"):
    if not (question or "").strip():
        st.warning("Enter a question.")
    else:
        q = question.strip()
        status_box.empty()
        answer_box.empty()
        try:
            with st.spinner("Agent thinking..."):
                ws = websocket.create_connection(WS_URL)
                try:
                    ws.send(json.dumps({"question": q}))
                    while True:
                        msg = ws.recv()
                        if not msg:
                            break
                        event = json.loads(msg)
                        ev = event.get("event", "")
                        data = event.get("data", "")
                        if ev == "rewrite":
                            status_box.caption(f"Rewrite: {data[:80]}..." if len(data) > 80 else f"Rewrite: {data}")
                        elif ev == "retrieval":
                            status_box.caption(data)
                        elif ev == "analysis":
                            status_box.caption(data)
                        elif ev == "answer":
                            answer_box.markdown(data)
                            break
                        elif ev == "error":
                            st.error(f"Error: {data}")
                            break
                finally:
                    ws.close()
        except Exception as e:
            st.error("Connection failed")
            status_box.caption(str(e))
