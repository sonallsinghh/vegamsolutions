# VegamSolutions — Agentic RAG

Minimal Agentic RAG app: document upload (PDF, txt, Excel), FastAPI backend, Streamlit UI. Service-layer structure; no embeddings or vector DB yet.

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
