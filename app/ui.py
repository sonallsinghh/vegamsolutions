# Run from project root: streamlit run app/ui.py
# UI talks to backend API (POST /upload, POST /query or WebSocket /ws/query). Chat history is stored on server by session_id.

import json
import os
import sys
import uuid
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

# Show documents already in the knowledge base (on every render)
try:
    r = requests.get(f"{API_BASE}/sources", timeout=10)
    if r.ok:
        sources = r.json().get("sources") or []
        if sources:
            st.caption("Documents in knowledge base:")
            for s in sources:
                st.caption(f"  • {s}")
        else:
            st.caption("No documents in knowledge base yet. Upload files below.")
    else:
        st.caption("Could not load document list.")
except requests.RequestException:
    st.caption("Backend not reachable — start the API first.")

# Clear knowledge base (with confirmation)
with st.expander("Clear knowledge base"):
    st.caption("Remove all documents from the vector store and delete all files in data/uploads/.")
    confirm_clear = st.checkbox("I understand this will remove all documents and uploaded files", key="confirm_clear")
    if st.button("Clear knowledge base", disabled=not confirm_clear, key="clear_kb"):
        try:
            r = requests.delete(f"{API_BASE}/sources", timeout=30)
            if r.ok:
                st.success("Knowledge base cleared.")
                st.rerun()
            else:
                st.error(f"Failed to clear: {r.status_code} — {r.text[:200]}")
        except requests.RequestException as e:
            st.error(f"Request failed: {e}")

uploaded_files = st.file_uploader(
    "Upload documents (.txt, .pdf, .xlsx, .xls)",
    type=["txt", "pdf", "xlsx", "xls"],
    accept_multiple_files=True,
)

# Upload only when user clicks "Upload" (not on every rerun / when clicking Submit)
if st.button("Upload", key="upload_btn") and uploaded_files:
    try:
        files = []
        for uf in uploaded_files:
            uf.seek(0)
            files.append(("files", (uf.name, uf.read())))
        r = requests.post(f"{API_BASE}/upload", files=files, timeout=30)
        if r.ok:
            data = r.json()
            st.success(f"Files uploaded successfully ({data.get('files_saved', 0)} saved)")
            st.rerun()
        else:
            st.error(f"Upload failed: {r.status_code} — {r.text[:200]}")
    except requests.RequestException as e:
        st.error(f"Upload failed: {e}")
elif uploaded_files:
    st.caption("Click **Upload** to add these files to the knowledge base.")

st.divider()
st.subheader("Chat")

# Session: one ID per conversation; server stores history by session_id
if "chat_session_id" not in st.session_state:
    st.session_state.chat_session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
# New chat: new session_id and clear local messages (server will have empty history for new id)
if st.button("New chat", key="new_chat"):
    st.session_state.chat_session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.rerun()

# Toggle REST vs WebSocket (selection stored in session_state.connection_mode)
st.radio(
    "Connection",
    options=["REST API", "WebSocket"],
    horizontal=True,
    key="connection_mode",
)
st.caption(f"Session: `{st.session_state.chat_session_id[:8]}...` (history stored on server)")

# Show previous messages (local display only)
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# New message from user
if prompt := st.chat_input("Ask a question about your documents"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    session_id = st.session_state.chat_session_id
    use_rest = st.session_state.get("connection_mode") == "REST API"

    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        answer_placeholder = st.empty()
        answer = ""
        try:
            if use_rest:
                with st.spinner("Thinking..."):
                    r = requests.post(
                        f"{API_BASE}/query",
                        json={"question": prompt, "session_id": session_id},
                        timeout=90,
                    )
                    if r.ok:
                        data = r.json()
                        answer = data.get("answer") or "No answer."
                        answer_placeholder.markdown(answer)
                        tools_used = data.get("tools_used") or []
                        if tools_used:
                            answer_placeholder.caption(f"Tools used: {', '.join(tools_used)}")
                    else:
                        answer = f"Error: {r.status_code} — {r.text[:200]}"
                        answer_placeholder.error(answer)
            else:
                with st.spinner("Thinking..."):
                    ws = websocket.create_connection(WS_URL)
                    try:
                        ws.send(json.dumps({"question": prompt, "session_id": session_id}))
                        while True:
                            msg_data = ws.recv()
                            if not msg_data:
                                break
                            event = json.loads(msg_data)
                            ev = event.get("event", "")
                            data = event.get("data", "")
                            if ev == "rewrite":
                                status_placeholder.caption(f"Rewrite: {data[:80]}..." if len(data) > 80 else f"Rewrite: {data}")
                            elif ev == "retrieval":
                                status_placeholder.caption(data)
                            elif ev == "analysis":
                                status_placeholder.caption(data)
                            elif ev == "answer":
                                answer = data
                                answer_placeholder.markdown(data)
                                break
                            elif ev == "error":
                                answer = f"Error: {data}"
                                answer_placeholder.error(data)
                                break
                    finally:
                        ws.close()
        except Exception as e:
            answer = f"Connection failed: {e}"
            answer_placeholder.error(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer or "No answer."})
    st.rerun()
