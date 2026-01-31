"""
API route aggregator: register endpoints; no logic — only delegate to handlers.
"""

import json
import logging

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app.api.handlers import handle_upload
from app.agent.graph import run_agent_agentic_stream
from app.core.session_store import append_message, get_history
from app.core.upload_db import clear_all as clear_upload_db, get_all_paths as get_upload_paths
from app.schemas.query import QueryRequest, QueryResponse
from app.schemas.upload import UploadResponse
from app.services.ingestion_service import clear_upload_dir, get_preview
from app.services.vector_store import clear_knowledge_base, list_sources

logger = logging.getLogger(__name__)
router = APIRouter()


# --- System ---

@router.get("/", tags=["system"])
def root():
    return {"status": "Agentic RAG backend running"}


@router.get("/health", tags=["system"])
def health():
    return {"ok": True}


# --- Ingestion ---

@router.get("/sources", tags=["ingestion"], summary="List documents in the knowledge base")
def get_sources() -> dict:
    """Return source names currently in the vector store (so the UI can show already-uploaded docs)."""
    try:
        sources = list_sources()
    except Exception as e:
        logger.warning("Failed to list sources: %s", e)
        sources = []
    return {"sources": sources}


@router.get(
    "/preview",
    tags=["ingestion"],
    summary="Preview how a document is processed (raw → cleaned → chunks)",
    description="Inspect data quality: returns raw excerpt, cleaned excerpt, and first N chunks for a source. Source must exist in data/uploads/.",
)
def preview_source(source: str = "") -> dict:
    """Return raw, cleaned, and chunked preview for the given source (filename). 404 if not found."""
    if not source or not source.strip():
        raise HTTPException(status_code=400, detail="Query parameter 'source' is required")
    data = get_preview(source.strip())
    if data is None:
        raise HTTPException(status_code=404, detail=f"Source not found or unreadable: {source.strip()!r}")
    return data


@router.get(
    "/uploads",
    tags=["ingestion"],
    summary="List stored upload file paths",
    description="Returns file paths stored in the SQLite upload DB (paths of uploaded files).",
)
def get_uploads() -> dict:
    """Return paths stored in the upload SQLite DB."""
    try:
        paths = get_upload_paths()
    except Exception as e:
        logger.warning("Failed to list upload paths: %s", e)
        paths = []
    return {"paths": paths}


@router.delete("/sources", tags=["ingestion"], summary="Clear the knowledge base")
def delete_sources() -> dict:
    """Drop the vector store collection, delete all files in data/uploads/, and clear upload DB."""
    try:
        clear_knowledge_base()
        files_removed = clear_upload_dir()
        clear_upload_db()
        return {"cleared": True, "files_removed": files_removed}
    except Exception as e:
        logger.exception("Failed to clear knowledge base")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post(
    "/upload",
    response_model=UploadResponse,
    tags=["ingestion"],
    summary="Upload and persist documents",
    description="Accept multiple .txt, .pdf, .xlsx, .xls files; save to data/uploads/; return count and paths. Rejects other types with 400. Returns 500 if saving fails.",
)
async def upload_and_persist_files(
    files: list[UploadFile] = File(..., description="One or more .txt, .pdf, .xlsx, or .xls files."),
) -> UploadResponse:
    return await handle_upload(files)


# --- Query (HTTP) ---

@router.post(
    "/query",
    response_model=QueryResponse,
    tags=["query"],
    summary="Query the RAG agent (sync)",
    description="Send a question; receive answer, iterations, chunks_used. 400 on invalid input, 500 on agent failure.",
)
def post_query(body: QueryRequest) -> QueryResponse:
    logger.info("[api:post_query] IN  question=%r session_id=%s", body.question, body.session_id)
    history = get_history(body.session_id)
    logger.info("[api:post_query] history_len=%d for session", len(history))
    try:
        answer = ""
        tools_used: list[str] = []
        for evt in run_agent_agentic_stream(body.question, history=history):
            event_type = evt.get("event", "")
            if event_type == "done":
                answer = evt.get("answer", "")
                tools_used = evt.get("tools_used", [])
                break
            if event_type == "error":
                raise RuntimeError(evt.get("message", "Agent error"))
        if not answer and not tools_used:
            answer = "I couldn't complete the request."
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("Agent failed")
        raise HTTPException(status_code=500, detail=str(e)) from e
    append_message(body.session_id, "user", body.question)
    append_message(body.session_id, "assistant", answer)
    logger.info("[api:post_query] OUT tools_used=%s answer_len=%d", tools_used, len(answer))
    return QueryResponse(
        answer=answer,
        iterations=0,
        chunks_used=0,
        tools_used=tools_used,
    )


def _sse_generator(question: str, session_id: str, history: list):
    """Yield Server-Sent Events for streaming agent response."""
    append_message(session_id, "user", question)
    try:
        for evt in run_agent_agentic_stream(question, history):
            event_type = evt.get("event", "")
            if event_type == "answer_delta":
                yield f"event: answer_delta\ndata: {json.dumps({'content': evt.get('content', '')})}\n\n"
            elif event_type == "tool":
                yield f"event: tool\ndata: {json.dumps({'name': evt.get('name', '')})}\n\n"
            elif event_type == "done":
                full_answer = evt.get("answer", "")
                tools_used = evt.get("tools_used", [])
                append_message(session_id, "assistant", full_answer)
                yield f"event: done\ndata: {json.dumps({'answer': full_answer, 'tools_used': tools_used})}\n\n"
            elif event_type == "error":
                yield f"event: error\ndata: {json.dumps({'message': evt.get('message', '')})}\n\n"
    except Exception as e:
        logger.exception("SSE stream failed")
        yield f"event: error\ndata: {json.dumps({'message': str(e)})}\n\n"


@router.post(
    "/query/stream",
    tags=["query"],
    summary="Query the RAG agent (SSE stream)",
    description="Stream the answer word-by-word via Server-Sent Events. Events: answer_delta, tool, done, error.",
)
def post_query_stream(body: QueryRequest) -> StreamingResponse:
    logger.info("[api:post_query_stream] IN  question=%r session_id=%s", body.question, body.session_id)
    history = get_history(body.session_id)
    try:
        return StreamingResponse(
            _sse_generator(body.question, body.session_id, history),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
