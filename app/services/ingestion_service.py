"""
Document ingestion: load, parse, and persist documents for RAG.

Responsibility: Orchestrate reading files (txt, pdf, excel), optional chunking,
and persistence. Called by the API layer; no HTTP or FastAPI here.
"""

import asyncio
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from app.core.config import ALLOWED_EXTENSIONS, CHUNK_OVERLAP, CHUNK_SIZE, UPLOAD_DIR_NAME
from app.ingest.loader import bytes_to_text
from app.services.text_processing import chunk_text, clean_text
from app.services.vector_store import store_chunks

logger = logging.getLogger(__name__)


class InvalidFileTypeError(Exception):
    """Raised when one or more files have disallowed extensions."""

    def __init__(self, invalid: list[str]) -> None:
        self.invalid = invalid
        super().__init__(f"Rejected: {', '.join(invalid)}")


@dataclass
class SaveUploadResult:
    """Result of saving uploaded files to disk."""

    files_saved: int
    paths: list[str]


def _project_root() -> Path:
    """Project root (VegamSolutions)."""
    return Path(__file__).resolve().parent.parent.parent


def _sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal (../). Returns safe basename."""
    if not filename or not filename.strip():
        return "unnamed"
    base = Path(filename).name
    safe = base.replace("..", "").replace("/", "").replace("\\", "")
    safe = re.sub(r"[^\w.\-]", "_", safe)
    return safe.strip() or "unnamed"


def save_uploaded_files(items: list[tuple[str, bytes]]) -> SaveUploadResult:
    """
    Validate, sanitize, and persist uploaded files to disk under data/uploads/.

    Args:
        items: List of (filename, raw_bytes) for each file.

    Returns:
        SaveUploadResult with files_saved count and relative paths (e.g. data/uploads/file.txt).

    Raises:
        InvalidFileTypeError: If any file has a disallowed extension (.txt, .pdf, .xlsx, .xls only).
        OSError: If creating the upload dir or writing a file fails.
    """
    if not items:
        raise InvalidFileTypeError([])

    root = _project_root() / UPLOAD_DIR_NAME
    root.mkdir(parents=True, exist_ok=True)

    invalid: list[str] = []
    saved_paths: list[str] = []
    seen_basenames: set[str] = set()

    for filename, content in items:
        ext = Path(filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            invalid.append(filename)
            continue

        safe_name = _sanitize_filename(filename)
        dest = root / safe_name
        while dest.exists() or safe_name in seen_basenames:
            stem, suf = dest.stem, dest.suffix
            if "_" in stem and stem.split("_")[-1].isdigit():
                n = int(stem.split("_")[-1]) + 1
                safe_name = f"{stem.rsplit('_', 1)[0]}_{n}{suf}"
            else:
                safe_name = f"{stem}_1{suf}"
            dest = root / safe_name
        seen_basenames.add(safe_name)

        dest_resolved = dest.resolve()
        if not str(dest_resolved).startswith(str(root.resolve())):
            invalid.append(filename)
            continue

        dest.write_bytes(content)
        rel_path = f"{UPLOAD_DIR_NAME}/{safe_name}"
        saved_paths.append(rel_path)

    if invalid:
        raise InvalidFileTypeError(invalid)

    return SaveUploadResult(files_saved=len(saved_paths), paths=saved_paths)


def _process_documents_sync(paths: list[str]) -> list[dict]:
    """
    Sync pipeline: read file → clean → chunk. Run in thread pool to avoid blocking event loop.
    """
    root = _project_root()
    all_chunks: list[dict] = []
    for path in paths:
        full_path = root / path
        if not full_path.exists():
            logger.warning("File not found: %s", path)
            continue
        try:
            raw = full_path.read_bytes()
            text = bytes_to_text(raw, full_path.name)
        except Exception as e:
            logger.warning("Failed to read %s: %s", path, e)
            continue
        cleaned = clean_text(text)
        chunks = chunk_text(cleaned, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        source_name = Path(path).name
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "text": chunk,
                "metadata": {"source": source_name, "chunk_id": i},
            })
        logger.info("File %s → %d chunks created", source_name, len(chunks))

    if all_chunks:
        try:
            store_chunks(all_chunks)
        except Exception as e:
            logger.warning("Failed to store chunks in Milvus: %s", e)

    return all_chunks


async def process_documents(paths: list[str]) -> list[dict]:
    """
    Read, clean, chunk each file; return structured chunks with metadata.

    Pipeline runs in thread pool so the event loop is not blocked by disk/CPU work.
    """
    return await asyncio.to_thread(_process_documents_sync, paths)
