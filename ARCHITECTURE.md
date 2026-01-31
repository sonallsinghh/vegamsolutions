# VegamSolutions — System Architecture

This document describes the high-level architecture of the Agentic RAG system: components, data flow, and external dependencies.

---

## 1. High-level architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              CLIENTS                                             │
│  Streamlit UI (app/ui.py)          External agents / MCP clients                │
└─────────────────────┬─────────────────────────────────┬─────────────────────────┘
                      │ HTTP                             │ HTTP (MCP tools)
                      ▼                                  ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         FastAPI APPLICATION (app/main.py)                         │
│  ┌─────────────────────────────────────┐  ┌─────────────────────────────────────┐│
│  │ API ROUTES (app/api/routes.py)      │  │ MCP ROUTER (app/mcp/server.py)       ││
│  │ /, /health, /sources, /upload,      │  │ POST /mcp/tools/search_documents      ││
│  │ /query, /query/stream, /preview,    │  │ POST /mcp/tools/list_sources          ││
│  │ /uploads, DELETE /sources           │  │ POST /mcp/tools/get_chunk             ││
│  └──────────────────┬──────────────────┘  │ POST /mcp/tools/system_stats         ││
│                      │                     └──────────────────┬──────────────────┘│
│                      │                                        │                    │
│  ┌───────────────────▼────────────────────────────────────────▼──────────────────┐│
│  │ HANDLERS (app/api/handlers.py)                                               ││
│  │ • handle_upload: validate files → save → kick off process_documents (async)  ││
│  └───────────────────┬─────────────────────────────────────────────────────────┘│
└───────────────────────┼─────────────────────────────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────────────────────────────┐
│                         CORE & ORCHESTRATION                                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────────┐ │
│  │ config.py       │  │ session_store.py │  │ upload_db.py (SQLite paths)      │ │
│  │ Env, constants  │  │ In-memory chat   │  │ Track uploaded file paths       │ │
│  └─────────────────┘  │ by session_id   │  └─────────────────────────────────┘ │
│                        └────────┬────────┘                                        │
│                                 │ used by /query                                   │
│  ┌──────────────────────────────▼───────────────────────────────────────────────┐ │
│  │ AGENT (app/agent/)                                                            │ │
│  │ • graph.py: LangGraph RAG (rewrite → retrieve → analyze → generate)          │ │
│  │             + agentic entry point (run_agent_agentic_stream) — tool-calling   │ │
│  │               loop                                                            │ │
│  │ • llm.py:   LLM: OpenAI → HF; chat_with_tools_stream                            │ │
│  │ • tools.py: Tool definitions (OpenAI format) + execute_tool (search_docs,     │ │
│  │             list_sources, get_chunk, system_stats, calculator, weather,       │ │
│  │             web_search)                                                       │ │
│  └──────────────────────────────┬───────────────────────────────────────────────┘ │
└─────────────────────────────────┼─────────────────────────────────────────────────┘
                                  │
┌─────────────────────────────────▼─────────────────────────────────────────────────┐
│                         SERVICES (app/services/)                                    │
│  ┌──────────────────────────┐  ┌──────────────────────────┐  ┌──────────────────┐ │
│  │ ingestion_service.py     │  │ retrieval_service.py      │  │ vector_store.py   │ │
│  │ • save_uploaded_files    │  │ • retrieve_context        │  │ • embed_texts (HF)│ │
│  │ • process_documents      │  │   - search_milvus         │  │ • get_milvus_     │ │
│  │ • get_preview            │  │   - _boost_by_keywords    │  │   client          │ │
│  │ • clear_upload_dir       │  │   - rerank_with_hf        │  │ • store_chunks    │ │
│  │ Uses: loader, text_      │  │ Uses: vector_store        │  │ • list_sources    │ │
│  │   processing,            │  │   (embed, search)        │  │ • get_chunk_by_id │ │
│  │   vector_store           │  │                           │  │ • clear_kb, stats │ │
│  └────────────┬─────────────┘  └────────────┬─────────────┘  └────────┬─────────┘ │
│               │                             │                          │          │
│  ┌────────────▼─────────────┐  ┌────────────▼─────────────┐            │          │
│  │ ingest/loader.py         │  │ text_processing.py       │            │          │
│  │ bytes_to_text (txt,     │  │ clean_text, chunk_text   │            │          │
│  │ pdf, xlsx, xls)          │  │                           │            │          │
│  └──────────────────────────┘  └──────────────────────────┘            │          │
└──────────────────────────────────────────────────────────────────────────┬─────────┘
                                                                           │
┌──────────────────────────────────────────────────────────────────────────▼─────────┐
│                         EXTERNAL DEPENDENCIES                                       │
│  • Milvus Cloud (Zilliz): vector store (collection "documents", COSINE, dim 384)   │
│  • Hugging Face: embeddings (all-MiniLM-L6-v2), rerank (bge-reranker-base),       │
│    optional chat (HF_LLM_MODEL)                                                    │
│  • OpenAI: agent LLM + tool-calling (gpt-4o-mini default)                         │
│  • Open-Meteo: weather (get_weather tool); optional ddgs: web_search               │
└───────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Component responsibilities

| Component | Responsibility |
|-----------|----------------|
| **API routes** | Register endpoints; delegate to handlers or agent; return HTTP/SSE. No business logic. |
| **Handlers** | Map HTTP types to service calls; map service exceptions to HTTP status. |
| **Agent (graph)** | RAG pipeline (LangGraph) and agentic loop: maintain messages, call LLM with tools, execute tools, loop until answer. |
| **Agent (llm)** | Single place for “call an LLM”: Ollama first, then OpenAI, then HF. Supports plain prompts and tool-calling (sync + stream). |
| **Agent (tools)** | Define tools (OpenAI function schema), implement `execute_tool` (search_documents → retrieval_service; list_sources, get_chunk, system_stats → vector_store; calculator, get_weather, web_search). |
| **Ingestion service** | Save uploaded files under `data/uploads/`, optionally enqueue processing; run pipeline: read → clean → chunk → store_chunks (vector_store). |
| **Retrieval service** | Embed query, search Milvus, keyword boost, HF rerank, return top chunks. |
| **Vector store** | Milvus client lifecycle, HF embeddings, insert/query/delete collection, list_sources, get_chunk_by_id, stats. |
| **Loader** | Convert file bytes to text by extension (txt, pdf, xlsx, xls). |
| **Text processing** | clean_text, chunk_text (size/overlap from config). |
| **Session store** | In-memory chat history keyed by session_id (used by /query). |
| **Upload DB** | SQLite list of uploaded file paths; updated on upload, cleared on DELETE /sources. |
| **MCP server** | HTTP tool endpoints that call retrieval_service and vector_store; same semantics as agent tools for external agents. |

---

## 3. Data flows

### 3.1 Ingestion (upload → vector store)

1. Client `POST /upload` with multipart files.
2. Handler validates and calls `save_uploaded_files` → files written to `data/uploads/`, paths recorded in upload_db.
3. Handler kicks off `process_documents(paths)` in background (asyncio task).
4. For each path: **loader** (bytes → text) → **text_processing** (clean, chunk) → chunks with metadata `{ source, chunk_id }`.
5. **vector_store**: `embed_texts` (HF) → Milvus insert (vector, text, source, chunk_id) → flush.

### 3.2 Query — agentic mode (default for /query)

1. Client `POST /query` with `{ question, session_id }`.
2. Routes load history from **session_store** by session_id.
3. **Agent** builds messages: system prompt + history + new question.
4. **Agent loop**: call **llm.chat_with_tools** (Ollama or OpenAI) with **tools** definitions; if LLM returns tool_calls, **execute_tool** for each (e.g. search_documents → **retrieval_service.retrieve_context**; list_sources/get_chunk/system_stats → **vector_store**; calculator/weather/web_search in tools.py).
5. Tool results appended to messages; repeat until LLM returns content only (no tool calls).
6. Final answer stored in session_store; response returns answer + tools_used.

### 3.3 Query — RAG pipeline (LangGraph)

Used internally by the graph (e.g. for non-agentic RAG path):

1. **rewrite_query**: HF (or configured) LLM rewrites query for retrieval (history-aware).
2. **retrieve_context**: retrieval_service (Milvus search → keyword boost → HF rerank).
3. **analyze_context**: LLM decides if context is sufficient (YES/NO).
4. If needs more and iterations < 2 → back to retrieve; else → **generate_answer** from chunks.
5. Answer returned; state holds chunks_used, iterations.

### 3.4 MCP tools (external agents)

- `POST /mcp/tools/search_documents`: body `{ query }` → retrieval_service.retrieve_context → return `{ results: [{ id, text, source }] }`.
- `POST /mcp/tools/list_sources` → vector_store.list_sources.
- `POST /mcp/tools/get_chunk`: body `{ id }` → vector_store.get_chunk_by_id.
- `POST /mcp/tools/system_stats` → vector_store.get_collection_stats.

No session or chat state; stateless tool calls.

---

## 4. Configuration and environment

- **config.py** loads `.env` and exposes: `MILVUS_URI`, `MILVUS_TOKEN`, `HF_API_KEY`, `OPENAI_API_KEY`, `OPENAI_LLM_MODEL`, `HF_LLM_MODEL`, chunk size/overlap, allowed extensions, upload dir, vector dimension, and app constants (timeouts, collection name, embed/rerank models, etc.).
- All external URLs and keys are read from config; no hardcoded secrets in application code.

---

## 5. Security and deployment notes

- Uploads: allowed extensions only; filenames sanitized; files stored under project-controlled `data/uploads/`.
- Session store: in-memory; not shared across processes; no persistence (restart clears history).
- API and MCP endpoints: no built-in auth in this design; assume reverse proxy or API gateway for TLS and authentication in production.
- Secrets: only in `.env`; `.env` not committed; `.env.example` documents variables.

---

## 6. Summary

| Aspect | Choice |
|--------|--------|
| **API** | FastAPI; routes + handlers; optional SSE for /query/stream. |
| **Agent** | LangGraph (RAG graph) + agentic tool-calling stream (OpenAI). |
| **Vector store** | Milvus Cloud; 384-d COSINE; HF embeddings. |
| **Retrieval** | Milvus