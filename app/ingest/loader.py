# Minimal document loader. No embeddings, no vector DB, no chunking.
# Supports .txt, .pdf, .xlsx, .xls.

import io
import os
from typing import List, Tuple

SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".xlsx", ".xls"}


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
            if ext == ".txt":
                texts.append(raw.decode("utf-8", errors="replace"))
            elif ext == ".pdf":
                texts.append(_read_pdf(raw))
            elif ext in (".xlsx", ".xls"):
                texts.append(_read_excel(raw))
        except Exception:
            unsupported.append(name)

    return texts, unsupported


def _read_pdf(raw: bytes) -> str:
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(raw))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _read_excel(raw: bytes) -> str:
    import pandas as pd
    df = pd.read_excel(io.BytesIO(raw), sheet_name=None, header=None)
    parts = []
    for sheet_name, sheet_df in df.items():
        parts.append(sheet_df.astype(str).to_csv(sep=" ", index=False, header=False))
    return "\n\n".join(parts)
