"""
API route aggregator: register endpoints; no logic — only delegate to handlers.
"""

from fastapi import APIRouter, File, UploadFile

from app.services.handlers import handle_upload
from app.schemas.upload import UploadResponse

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
