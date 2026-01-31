# Run from project root: streamlit run app/ui.py
# UI talks to backend API (POST /upload, POST /query, POST /query/stream for SSE). Chat history is stored on server by session_id.

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

# Backend config
API_BASE = os.environ.get("API_BASE", "http://localhost:8000")

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

# Inspect data quality: raw → cleaned → chunks
with st.expander("Inspect data quality"):
    st.caption("Preview how a document is processed: raw text, cleaned text, and chunks that go into the index.")
    try:
        r_sources = requests.get(f"{API_BASE}/sources", timeout=10)
        preview_sources = (r_sources.json().get("sources") or []) if r_sources.ok else []
    except requests.RequestException:
        preview_sources = []
    if not preview_sources:
        st.caption("No documents in the knowledge base. Upload files first.")
    else:
        selected = st.selectbox("Choose a document", preview_sources, key="preview_source")
        if st.button("Preview", key="preview_btn") and selected:
            try:
                r = requests.get(f"{API_BASE}/preview", params={"source": selected}, timeout=30)
                if r.ok:
                    data = r.json()
                    st.metric("Chunks", data.get("chunk_count", 0))
                    st.caption(f"Raw length: {data.get('raw_len', 0)} chars → Cleaned: {data.get('cleaned_len', 0)} chars")
                    with st.expander("Raw text (excerpt)", expanded=False):
                        st.text_area("raw", value=data.get("raw_excerpt", ""), height=200, key="preview_raw", disabled=True)
                    with st.expander("Cleaned text (excerpt)", expanded=False):
                        st.text_area("cleaned", value=data.get("cleaned_excerpt", ""), height=200, key="preview_cleaned", disabled=True)
                    st.subheader("Chunks (first 10)")
                    for c in data.get("chunks", []):
                        st.text_area(f"Chunk {c.get('chunk_id', 0)}", value=c.get("text", ""), height=120, key=f"preview_chunk_{c.get('chunk_id')}", disabled=True)
                else:
                    st.error(f"Preview failed: {r.status_code} — {r.text[:200]}")
            except Exception as e:
                st.error(f"Request failed: {e}")

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

#st.caption(f"Session: `{st.session_state.chat_session_id[:8]}...` (history stored on server)")

# Show previous messages (local display only)
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# If we just submitted a query, show it and then show "Thinking..." while waiting for response
if st.session_state.get("pending_query"):
    prompt = st.session_state.pending_query
    session_id = st.session_state.chat_session_id
    with st.chat_message("assistant"):
        thinking_placeholder = st.empty()
        thinking_placeholder.caption("Thinking...")
        answer_placeholder = st.empty()
        tools_caption = st.empty()
        answer = ""
        tools_used: list[str] = []
        try:
            r = requests.post(
                f"{API_BASE}/query/stream",
                json={"question": prompt, "session_id": session_id},
                stream=True,
                timeout=90,
            )
            if not r.ok:
                answer = f"Error: {r.status_code} — {r.text[:200]}"
                thinking_placeholder.empty()
                answer_placeholder.error(answer)
            else:
                current_event = None
                accumulated = []
                for line in r.iter_lines(decode_unicode=True):
                    if line is None:
                        continue
                    if line.startswith("event:"):
                        current_event = line[6:].strip()
                    elif line.startswith("data:") and current_event:
                        try:
                            data = json.loads(line[5:].strip())
                        except json.JSONDecodeError:
                            data = {}
                        if current_event == "answer_delta":
                            content = data.get("content", "")
                            if content:
                                accumulated.append(content)
                                thinking_placeholder.empty()
                                answer_placeholder.markdown("".join(accumulated))
                        elif current_event == "tool":
                            name = data.get("name", "")
                            if name:
                                tools_used.append(name)
                                thinking_placeholder.caption("Thinking...")
                                tools_caption.caption(f"Tools used: {', '.join(tools_used)}")
                        elif current_event == "done":
                            answer = data.get("answer", "") or "".join(accumulated)
                            tools_used = data.get("tools_used", []) or tools_used
                            thinking_placeholder.empty()
                            if answer:
                                answer_placeholder.markdown(answer)
                            if tools_used:
                                tools_caption.caption(f"Tools used: {', '.join(tools_used)}")
                        elif current_event == "error":
                            msg = data.get("message", "Unknown error")
                            thinking_placeholder.empty()
                            answer_placeholder.error(msg)
                            answer = msg
                answer = answer or "".join(accumulated) or "No answer."
        except Exception as e:
            answer = f"Connection failed: {e}"
            thinking_placeholder.empty()
            answer_placeholder.error(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer or "No answer."})
    del st.session_state["pending_query"]
    st.rerun()

# New message from user: show it immediately, then rerun so "Thinking..." appears
if prompt := st.chat_input("Ask a question about your documents, weather of a place, or calculate or web search"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.pending_query = prompt
    st.rerun()
