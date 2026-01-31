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

```bash
uvicorn app.main:app --reload
```

**Streamlit UI:**

```bash
streamlit run app/ui.py
```

**CLI:**

```bash
python -m app.main
```

## MCP tool server usage

The backend exposes retrieval as an MCP-compatible tool so external agents can call it over HTTP.

- **Endpoint:** `POST /mcp/tools/search_documents`
- **Request body:** `{"query": "your search question"}`
- **Response:** `{"results": [{"text": "...", "source": "..."}, ...]}`

Example with `curl`:

```bash
curl -X POST http://localhost:8000/mcp/tools/search_documents \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the main topic?"}'
```

Each item in `results` is a chunk from the document store (semantic search + rerank). Use this endpoint from MCP clients or any agent that expects a standardized tool interface.
