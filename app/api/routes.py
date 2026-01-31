"""
API route aggregator: register endpoints; no logic — only delegate to handlers.
"""

import logging

from fastapi import APIRouter, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect

from app.api.handlers import handle_upload
from app.agent.graph import run_agent, run_agent_stream
from app.schemas.query import QueryRequest, QueryResponse
from app.schemas.upload import UploadResponse

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
    logger.info("Query received")
    try:
        result = run_agent(body.question)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("Agent failed")
        raise HTTPException(status_code=500, detail=str(e)) from e
    logger.info("Agent completed in %d iterations", result["iterations"])
    return QueryResponse(
        answer=result["answer"],
        iterations=result["iterations"],
        chunks_used=result["chunks_used"],
    )


# --- Query (WebSocket streaming) ---

@router.websocket("/ws/query")
async def ws_query(websocket: WebSocket) -> None:
    await websocket.accept()
    logger.info("WebSocket session started")
    try:
        data = await websocket.receive_json()
        question = (data.get("question") or "").strip()
        if not question:
            await websocket.send_json({"event": "error", "data": "question is required"})
            return
        for event in run_agent_stream(question):
            await websocket.send_json(event)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.exception("WebSocket query failed")
        try:
            await websocket.send_json({"event": "error", "data": str(e)})
        except Exception:
            pass
    finally:
        logger.info("WebSocket session closed")
