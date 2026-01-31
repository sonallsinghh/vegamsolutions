"""
API handlers: read request data (e.g. UploadFile), call services, map results/errors to HTTP.

Responsibility: Bridge HTTP types and services. Marshalling and exception-to-HTTP mapping.
Lives in the API layer so services stay free of FastAPI/HTTP types.
"""

import asyncio

from fastapi import HTTPException, UploadFile

from app.schemas.upload import UploadResponse
from app.services.ingestion_service import InvalidFileTypeError, process_documents, save_uploaded_files


async def handle_upload(files: list[UploadFile]) -> UploadResponse:
    """
    Read uploaded files, call ingestion service, map service errors to HTTP 400/500.
    Kicks off process_documents in background so the response returns immediately.
    """
    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required.")

    items: list[tuple[str, bytes]] = []
    for upload in files:
        filename = upload.filename or ""
        content = await upload.read()
        items.append((filename, content))

    try:
        result = save_uploaded_files(items)
    except InvalidFileTypeError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Only .txt, .pdf, .xlsx, .xls are allowed. Rejected: {', '.join(e.invalid)}",
        ) from e
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Failed to save files: {e!s}") from e

    if result.paths:
        asyncio.create_task(process_documents(result.paths))

    return UploadResponse(files_saved=result.files_saved, paths=result.paths)
