# Minimal document loader. No embeddings, no vector DB, no chunking.
# Supports .txt, .pdf, .xlsx, .xls. Single place for "file/bytes → text".

import io
import os
from pathlib import Path
from typing import List, Tuple

from app.core.config import ALLOWED_EXTENSIONS

SUPPORTED_EXTENSIONS = ALLOWED_EXTENSIONS


def bytes_to_text(raw: bytes, filename: str) -> str:
    """
    Convert raw file bytes to text by extension. Single source of truth for
    .txt, .pdf, .xlsx, .xls parsing. Used by Streamlit loader and ingestion pipeline.
    """
    ext = Path(filename).suffix.lower() if filename else ""
    if ext == ".txt" or not ext:
        return raw.decode("utf-8", errors="replace")
    if ext == ".pdf":
        return _read_pdf(raw)
    if ext in (".xlsx", ".xls"):
        return _read_excel(raw)
    return raw.decode("utf-8", errors="replace")


def _read_pdf(raw: bytes) -> str:
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(raw))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _read_excel(raw: bytes) -> str:
    import pandas as pd
    df = pd.read_excel(io.BytesIO(raw), sheet_name=None, header=None)
    parts = []
    for sheet_df in df.values():
        parts.append(sheet_df.astype(str).to_csv(sep=" ", index=False, header=False))
    return "\n\n".join(parts)


def load_files(files) -> Tuple[List[str], List[str]]:
    """
    Read uploaded Streamlit files as text.
    Returns (list of document texts, list of unsupported filenames).
    """
    if not files:
        return [], []

    texts: List[str] = []
    unsupported: List[str] = []

    for f in files:
        name = getattr(f, "name", str(f))
        ext = os.path.splitext(name)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            unsupported.append(name)
            continue
        f.seek(0)
        raw = f.read()
        try:
            texts.append(bytes_to_text(raw, name))
        except Exception:
            unsupported.append(name)

    return texts, unsupported
