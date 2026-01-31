# VegamSolutions — Agentic RAG

An **Agentic RAG** (Retrieval-Augmented Generation) application: upload documents (PDF, TXT, Excel), ingest them into a vector store, and query via an AI agent that can use tools (semantic search, calculator, weather, web search) and maintain conversation history.

## Features

- **Document ingestion**: Upload `.txt`, `.pdf`, `.xlsx`, `.xls`; automatic chunking, cleaning, embedding (Hugging Face), and storage in **Milvus Cloud**.
- **Agentic query**: LangGraph-based agent with **tool-calling** (OpenAI): `search_documents`, `list_sources`, `get_chunk`, `system_stats`, `current_date`, `calculator`, `get_weather`, `web_search`.
- **RAG pipeline** (optional): Query rewrite → semantic retrieval (Milvus) → keyword boost → HF rerank → context analysis → answer generation; up to 2 retrieval passes.
- **Session-aware chat**: Server-side chat history by `session_id`; supports multi-turn context.
- **Streaming**: SSE streaming for agent responses (`/query/stream`).
- **MCP tool server**: HTTP endpoints exposing retrieval and KB introspection for external agents (`/mcp/tools/*`).
- **Streamlit UI**: Upload files, clear KB, chat with the agent (sync or stream).

## Tech stack

| Layer | Technology |
|-------|------------|
| API | FastAPI, uvicorn |
| UI | Streamlit |
| Agent | LangGraph, tool-calling (OpenAI) |
| Embeddings | Hugging Face Inference API (`sentence-transformers/all-MiniLM-L6-v2`) |
| Rerank | Hugging Face (`BAAI/bge-reranker-base`) |
| Vector DB | Milvus Cloud (Zilliz) |
| LLM (simple prompts) | OpenAI → Hugging Face |
| LLM (tool-calling) | OpenAI |

## Environment setup

1. **Copy env and set secrets**

   ```bash
   cp .env.example .env
   ```

   Edit `.env` and set:

   | Variable | Description |
   |----------|-------------|
   | `MILVUS_URI` | Milvus Cloud cluster URI (e.g. `https://xxx.api.gcp-us-west1.zillizcloud.com:19530`) |
   | `MILVUS_TOKEN` | Milvus Cloud API token or `username:password` |
   | `HF_API_KEY` | Hugging Face API key (embeddings, rerank, optional LLM fallback) |
   | `OPENAI_API_KEY` | OpenAI API key (required for agentic tool-calling) |
   | `OPENAI_LLM_MODEL` | Optional; default `gpt-4o-mini` |
   | `HF_LLM_MODEL` | Optional; used when OpenAI is not set (fallback LLM) |

   **Do not commit `.env`** — it is in `.gitignore`. Use `.env.example` as a template.

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   Key packages: `fastapi`, `uvicorn`, `streamlit`, `langgraph`, `pymilvus`, `python-dotenv`, `openai`, `httpx`, `pypdf`, `openpyxl`, `pandas`, `pytest`. Optional: `ddgs` for `web_search` tool.

3. **Milvus**: On first run, the app creates the `documents` collection if missing. Ensure `MILVUS_URI` and `MILVUS_TOKEN` are set or connection will fail.

## SQLite upload DB

Uploaded file paths are stored in a local SQLite database so the app can list them via `GET /uploads` and clear them when the KB is cleared.

| Item | Value |
|------|--------|
| **Path** | `data/uploads.db` (under project root) |
| **Table** | `uploads` |
| **Columns** | `id` (INTEGER PK), `path` (TEXT), `created_at` (TEXT) |

The file is created on first upload. To view it:

```bash
sqlite3 data/uploads.db "SELECT * FROM uploads;"
```

Or open `data/uploads.db` in a SQLite GUI (e.g. [DB Browser for SQLite](https://sqlitebrowser.org/)).

**Seed DB for demos/tests:** From project root, run:

```bash
python scripts/seed_upload_db.py          # append seed paths
python scripts/seed_upload_db.py --reset # clear then seed
```

Edit `scripts/seed_upload_db.py` and change `SEED_PATHS` to match your demo files under `data/uploads/`.

## Project structure

```
VegamSolutions/
├── app/
│   ├── main.py                 # FastAPI app: uvicorn app.main:app --reload
│   ├── ui.py                   # Streamlit UI: streamlit run app/ui.py
│   ├── api/
│   │   ├── routes.py           # All HTTP routes (system, ingestion, query, stream)
│   │   └── handlers.py         # Upload handler (HTTP → services)
│   ├── agent/
│   │   ├── graph.py            # LangGraph: rewrite → retrieve → analyze → generate (RAG); agentic stream
│   │   ├── llm.py              # LLM: OpenAI → HF (prompts + tool-calling stream)
│   │   └── tools.py            # Tool definitions + execution (search_documents, calculator, weather, etc.)
│   ├── core/
│   │   ├── config.py           # Env and app constants
│   │   ├── session_store.py    # In-memory chat history by session_id
│   │   └── upload_db.py        # SQLite store of uploaded file paths
│   ├── ingest/
│   │   └── loader.py           # File → text (txt, pdf, xlsx, xls)
│   ├── llm/
│   │   └── __init__.py         # (Agent LLM in app.agent.llm)
│   ├── mcp/
│   │   └── server.py           # MCP-style HTTP tools: search_documents, list_sources, get_chunk, system_stats
│   ├── schemas/
│   │   ├── query.py            # QueryRequest, QueryResponse
│   │   └── upload.py           # UploadResponse
│   └── services/
│       ├── ingestion_service.py # Save files, clean, chunk, store in Milvus
│       ├── retrieval_service.py # Milvus search → keyword boost → HF rerank
│       ├── vector_store.py      # Milvus client, HF embeddings, collection ops
│       ├── text_processing.py   # clean_text, chunk_text
│       └── agent_service.py    # (if used) orchestration around agent
├── data/
│   ├── uploads/                # Persisted uploaded files
│   └── uploads.db              # SQLite DB of upload paths (created on first upload)
├── scripts/
│   └── seed_upload_db.py       # Seed uploads DB for demos/tests
├── tests/
│   ├── test_text_processing.py
│   └── test_mcp_integration.py
├── requirements.txt
├── .env.example
└── README.md
```

## Run

From the **VegamSolutions** directory:

```bash
cd VegamSolutions
pip install -r requirements.txt
```

**API (FastAPI)**

```bash
uvicorn app.main:app --reload
```

API base URL: `http://localhost:8000`. OpenAPI docs: `http://localhost:8000/docs`.

**Streamlit UI**

```bash
streamlit run app/ui.py
```

Set `API_BASE` if the API runs elsewhere (e.g. `export API_BASE=http://localhost:8000`).

**Tests**

```bash
pytest tests/ -v
```

Tests cover text processing and MCP integration (e.g. `POST /mcp/tools/search_documents` with mocked retrieval).

## API overview

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health / status |
| GET | `/health` | Health check |
| GET | `/sources` | List document sources in the knowledge base |
| GET | `/preview?source=<name>` | Preview raw → cleaned → chunks for a source |
| GET | `/uploads` | List stored upload paths (SQLite) |
| DELETE | `/sources` | Clear KB (drop collection, delete uploads, clear DB) |
| POST | `/upload` | Upload files (multipart); ingest in background |
| POST | `/query` | Query agent (sync); body: `question`, `session_id` |
| POST | `/query/stream` | Query agent with SSE stream |

## MCP tool server

| Tool | Endpoint | Description |
|------|----------|-------------|
| **search_documents** | `POST /mcp/tools/search_documents` | Semantic search + rerank. Body: `{"query": "..."}`. Returns `{"results": [{"id", "text", "source"}, ...]}`. |
| **list_sources** | `POST /mcp/tools/list_sources` | List source names in the KB. Body: `{}`. |
| **get_chunk** | `POST /mcp/tools/get_chunk` | Get chunk by id. Body: `{"id": 123}`. |
| **system_stats** | `POST /mcp/tools/system_stats` | KB stats: collection name, total chunks, sources. |

Example:

```bash
curl -X POST http://localhost:8000/mcp/tools/search_documents \
  -H "Content-Type: application/json" \
  -d '{"query": "leave policy"}'
```

## Query flow (agentic mode)

1. Client sends `POST /query` with `question` and `session_id`.
2. Server loads chat history for `session_id`.
3. Agent runs in **agentic** mode: system prompt + history + question; LLM (Ollama or OpenAI) may call tools (`search_documents`, `list_sources`, `get_chunk`, `calculator`, `get_weather`, `web_search`, etc.).
4. Tool results are fed back; loop continues until the LLM returns a final answer (no more tool calls).
5. Answer and history are stored; response returns `answer` and `tools_used`.

For streaming, use `POST /query/stream`; events include `answer_delta`, `tool`, `done`, `error`.

## Error handling

When a required dependency is missing or unreachable, the API returns a clear response instead of crashing:

- **503 Service Unavailable** – e.g. vector store or embeddings not configured. Example: if `MILVUS_URI` or `MILVUS_TOKEN` is empty in `.env`, the first query that triggers a search (e.g. asking a question) will fail with 503 and a message like *"Vector store is not configured. Please set MILVUS_URI and MILVUS_TOKEN in .env."* Similarly, a missing `HF_API_KEY` yields a 503 when embeddings are needed.
- **500** – Other agent or server errors (logged; detail may be returned).
- **400** – Invalid input (e.g. empty question).

So if you forget to set Milvus or HF in `.env`, the app handles it with a 503 and a user-facing message.

## License and attribution

Use `.env.example` as the single reference for required and optional environment variables. Do not commit `.env`.
