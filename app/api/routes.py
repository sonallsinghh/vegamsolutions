"""
API route aggregator: register endpoints; no logic — only delegate to handlers.
"""

import logging

from fastapi import APIRouter, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect

from app.api.handlers import handle_upload
from app.agent.graph import run_agent_agentic, run_agent_stream
from app.core.session_store import append_message, get_history
from app.schemas.query import QueryRequest, QueryResponse
from app.schemas.upload import UploadResponse
from app.services.ingestion_service import clear_upload_dir
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


@router.delete("/sources", tags=["ingestion"], summary="Clear the knowledge base")
def delete_sources() -> dict:
    """Drop the vector store collection and delete all files in data/uploads/."""
    try:
        clear_knowledge_base()
        files_removed = clear_upload_dir()
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
    logger.info("[api:post_query] IN  question=%r session_id=%s agentic=%s", body.question, body.session_id, body.agentic)
    history = get_history(body.session_id)
    logger.info("[api:post_query] history_len=%d for session", len(history))
    try:
        if body.agentic:
            result = run_agent_agentic(body.question, history=history)
            append_message(body.session_id, "user", body.question)
            append_message(body.session_id, "assistant", result["answer"])
            logger.info("[api:post_query] OUT agentic tools_used=%s answer_len=%d", result.get("tools_used", []), len(result["answer"]))
            return QueryResponse(
                answer=result["answer"],
                iterations=0,
                chunks_used=0,
                tools_used=result.get("tools_used", []),
            )
        result = run_agent(body.question, history=history)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("Agent failed")
        raise HTTPException(status_code=500, detail=str(e)) from e
    append_message(body.session_id, "user", body.question)
    append_message(body.session_id, "assistant", result["answer"])
    logger.info("[api:post_query] OUT iterations=%d chunks_used=%d answer_len=%d", result["iterations"], result["chunks_used"], len(result["answer"]))
    logger.info("[api:post_query] OUT answer=%r", result["answer"])
    return QueryResponse(
        answer=result["answer"],
        iterations=result["iterations"],
        chunks_used=result["chunks_used"],
    )


# --- Query (WebSocket streaming) ---

@router.websocket("/ws/query")
async def ws_query(websocket: WebSocket) -> None:
    await websocket.accept()
    logger.info("[api:ws_query] WebSocket session started")
    try:
        data = await websocket.receive_json()
        question = (data.get("question") or "").strip()
        session_id = (data.get("session_id") or "").strip()
        logger.info("[api:ws_query] IN  question=%r session_id=%s", question, session_id)
        if not question:
            await websocket.send_json({"event": "error", "data": "question is required"})
            return
        if not session_id:
            await websocket.send_json({"event": "error", "data": "session_id is required"})
            return
        history = get_history(session_id)
        answer = ""
        for event in run_agent_stream(question, history=history):
            await websocket.send_json(event)
            if event.get("event") == "answer":
                answer = event.get("data") or ""
        if answer:
            append_message(session_id, "user", question)
            append_message(session_id, "assistant", answer)
            logger.info("[api:ws_query] OUT answer_len=%d appended to session", len(answer))
    except WebSocketDisconnect:
        logger.info("[api:ws_query] client disconnected")
    except Exception as e:
        logger.exception("[api:ws_query] WebSocket query failed")
        try:
            await websocket.send_json({"event": "error", "data": str(e)})
        except Exception:
            pass
    finally:
        logger.info("[api:ws_query] WebSocket session closed")
