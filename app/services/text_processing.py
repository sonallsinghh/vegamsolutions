"""
Text processing for RAG: cleaning and chunking.

Cleaning reduces noise and encoding inconsistencies so embeddings and retrieval
focus on content. Chunk quality directly impacts retrieval accuracy.
"""

import re
import unicodedata


def clean_text(text: str) -> str:
    """
    Normalize and clean raw document text for RAG.

    Cleaning matters in RAG because: noisy or inconsistent text (extra spaces,
    duplicate lines, mixed unicode) degrades embedding quality and adds
    irrelevant matches. Normalized text yields more consistent embeddings and
    better retrieval.
    """
    if not text or not text.strip():
        return ""
    text = unicodedata.normalize("NFKC", text)
    lines = [line.strip() for line in text.splitlines()]
    deduped: list[str] = []
    for line in lines:
        if deduped and deduped[-1] == line:
            continue
        deduped.append(line)
    result: list[str] = []
    for line in deduped:
        if line == "":
            if result and result[-1] != "":
                result.append("")
        else:
            result.append(line)
    return "\n".join(result).strip()


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Split text into overlapping chunks with paragraph/sentence awareness.

    Chunk quality directly impacts retrieval accuracy. Paragraph-aware
    splitting and preserving sentence boundaries keep semantic units intact;
    overlap provides context across chunk boundaries; avoiding mid-word cuts
    keeps tokens meaningful.
    """
    if not text or not text.strip():
        return []
    text = text.strip()
    if len(text) <= chunk_size:
        return [text] if text else []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        sentences = re.split(r"\s+", text)
        sentences = [s for s in sentences if s]

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sent in sentences:
        add_len = len(sent) + (1 if current else 0)
        if current_len + add_len <= chunk_size:
            current.append(sent)
            current_len += add_len
        else:
            if current:
                chunk = " ".join(current)
                chunks.append(chunk)
                overlap_parts: list[str] = []
                overlap_len = 0
                for s in reversed(current):
                    if overlap_len + len(s) + 1 <= overlap:
                        overlap_parts.append(s)
                        overlap_len += len(s) + 1
                    else:
                        break
                overlap_parts.reverse()
                current = overlap_parts if overlap_parts else []
                current_len = sum(len(s) for s in current) + max(0, len(current) - 1)
            if len(sent) > chunk_size:
                words = sent.split()
                for w in words:
                    add_len = len(w) + (1 if current else 0)
                    if current_len + add_len <= chunk_size:
                        current.append(w)
                        current_len += add_len
                    else:
                        if current:
                            chunk = " ".join(current)
                            chunks.append(chunk)
                            overlap_parts = []
                            overlap_len = 0
                            for x in reversed(current):
                                if overlap_len + len(x) + 1 <= overlap:
                                    overlap_parts.append(x)
                                    overlap_len += len(x) + 1
                                else:
                                    break
                            overlap_parts.reverse()
                            current = overlap_parts if overlap_parts else []
                            current_len = sum(len(x) for x in current) + max(0, len(current) - 1)
                        current.append(w)
                        current_len += len(w) + 1
            else:
                current.append(sent)
                current_len += len(sent) + 1

    if current:
        chunks.append(" ".join(current))

    return chunks
