# VegamSolutions — Agentic RAG

Minimal Agentic RAG app: document upload (PDF, txt, Excel), FastAPI backend, Streamlit UI. LangGraph + Milvus Cloud skeleton; no embeddings or vector inserts yet.

## Environment setup

1. **Copy env file and fill in secrets**

   ```bash
   cp .env.example .env
   ```

   Edit `.env` and set:

   - **MILVUS_URI** — Your Milvus Cloud cluster endpoint (e.g. `https://xxx.api.gcp-us-west1.zillizcloud.com:19530`).
   - **MILVUS_TOKEN** — Milvus Cloud API key or `username:password`.
   - **HF_API_KEY** — Hugging Face API key (for embeddings / inference later).

2. **Do not commit `.env`** — It is listed in `.gitignore`. Use `.env.example` as a template for others.

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   Key deps: `pymilvus` (Milvus Cloud), `langgraph`, `python-dotenv`, `fastapi`, `uvicorn`.

4. **Verify Milvus** — On first run, the app will connect to Milvus and create collection `documents` if missing. Ensure `MILVUS_URI` and `MILVUS_TOKEN` are set or connection will fail.

## Project structure

```
VegamSolutions/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app entry: uvicorn app.main:app --reload
│   ├── ui.py                # Streamlit UI: streamlit run app/ui.py
│   ├── api/                 # HTTP layer (routes only)
│   │   ├── __init__.py
│   │   ├── routes.py        # Aggregates all API routes
│   │   └── upload.py        # POST /upload
│   ├── core/                # Config and shared utilities
│   │   ├── __init__.py
│   │   └── config.py         # App config (placeholder)
│   ├── services/            # Business logic (no HTTP)
│   │   ├── __init__.py
│   │   ├── ingestion_service.py
│   │   ├── retrieval_service.py
│   │   └── agent_service.py
│   ├── schemas/             # Pydantic request/response models
│   │   ├── __init__.py
│   │   └── upload.py
│   └── ingest/              # Document loading (used by UI / future services)
│       ├── __init__.py
│       └── loader.py
├── data/
│   └── uploads/             # Persisted uploads
├── tests/
│   └── __init__.py
├── requirements.txt
└── README.md
```

## Run

From project root:

```bash
pip install -r requirements.txt
```

**API (FastAPI + uvicorn):**  
Run from the **VegamSolutions** directory so the correct app (with `/upload`, `/query`, MCP) is loaded:

```bash
cd VegamSolutions
uvicorn app.main:app --reload
```

If you run `uvicorn app.main:app` from the parent **First500** directory, the wrong app (no `/upload`) will run and uploads will return 404.

**Streamlit UI:**

```bash
streamlit run app/ui.py
```

**CLI:**

```bash
python -m app.main
```

**Tests:**

```bash
pytest tests/ -v
```

Unit tests cover `clean_text` and `chunk_text`; integration tests hit `POST /mcp/tools/search_documents` with a mocked retrieval layer (no Milvus/HF required).

## MCP tool server usage

The backend exposes retrieval and knowledge-base introspection as MCP-compatible tools over HTTP.

| Tool | Endpoint | Description |
|------|----------|-------------|
| **search_documents** | `POST /mcp/tools/search_documents` | Semantic search + rerank. Body: `{"query": "..."}`. Returns `{"results": [{"id", "text", "source"}, ...]}`. Each `id` is the Milvus primary key so the agent can call **get_chunk**(id). |
| **list_sources** | `POST /mcp/tools/list_sources` | Document introspection: list source names in the KB. Body: `{}`. Returns `{"sources": ["doc1.txt", ...]}`. |
| **get_chunk** | `POST /mcp/tools/get_chunk` | Metadata awareness: fetch chunk by Milvus entity id. Body: `{"id": 123}`. Returns `{"chunk": {"id", "text", "source", "chunk_id"}}` or `{"chunk": null}`. |
| **system_stats** | `POST /mcp/tools/system_stats` | System observability: KB status. Body: `{}`. Returns `{"collection_name", "total_chunks", "source_count", "sources": [...]}`. |

Example (search):

```bash
curl -X POST http://localhost:8000/mcp/tools/search_documents \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the main topic?"}'
```

Example (list sources):

```bash
curl -X POST http://localhost:8000/mcp/tools/list_sources -H "Content-Type: application/json" -d '{}'
```

Example (system stats):

```bash
curl -X POST http://localhost:8000/mcp/tools/system_stats -H "Content-Type: application/json" -d '{}'
```
