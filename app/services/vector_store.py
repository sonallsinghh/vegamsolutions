"""
Vector store client: Milvus Cloud connection, embeddings (HF Inference API), and chunk storage.

Responsibility: Connect to Milvus, embed texts via all-MiniLM-L6-v2, store chunks with metadata.
"""

import logging
from typing import Any

import httpx

from app.core.config import HF_API_KEY, MILVUS_TOKEN, MILVUS_URI, VECTOR_DIM

logger = logging.getLogger(__name__)

COLLECTION_NAME = "documents"

# Hugging Face Inference API: sentence-transformers/all-MiniLM-L6-v2 (dim 384)
HF_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HF_API_URL_ROUTER = (
    "https://router.huggingface.co/hf-inference/models/"
    f"{HF_EMBED_MODEL}/pipeline/feature-extraction"
)
HF_API_URL_STANDARD = f"https://api-inference.huggingface.co/models/{HF_EMBED_MODEL}"
API_TIMEOUT = 30.0
EMBED_BATCH_SIZE = 32


def embed_texts(texts: list[str], batch_size: int = EMBED_BATCH_SIZE) -> list[list[float]]:
    """
    Batch embed texts using Hugging Face Inference API (all-MiniLM-L6-v2).

    Batch embeddings reduce latency and improve throughput.
    Returns list of 384-dim vectors (normalized for cosine similarity).
    """
    if not texts:
        return []
    if not HF_API_KEY:
        raise ValueError(
            "HF_API_KEY must be set in .env. Get a token from https://huggingface.co/settings/tokens"
        )

    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json",
    }
    all_embeddings: list[list[float]] = []

    with httpx.Client(timeout=API_TIMEOUT) as client:
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            payload = {"inputs": batch, "options": {"wait_for_model": True}}
            api_urls = [HF_API_URL_ROUTER, HF_API_URL_STANDARD]
            response = None
            last_error: str | None = None

            for api_url in api_urls:
                try:
                    response = client.post(api_url, json=payload, headers=headers)
                    if response.status_code == 200:
                        break
                    if response.status_code == 403 and api_url == HF_API_URL_ROUTER:
                        last_error = response.text
                        continue
                    break
                except Exception as e:
                    last_error = str(e)
                    if api_url == api_urls[-1]:
                        raise
                    continue

            if response is None or response.status_code != 200:
                msg = response.text if response else last_error
                if response and response.status_code == 503:
                    raise RuntimeError(f"HF model is loading. Retry later. {msg}")
                if response and response.status_code == 401:
                    raise ValueError(
                        "Invalid HF API key. Check HF_API_KEY at https://huggingface.co/settings/tokens"
                    )
                if response and response.status_code == 403:
                    raise ValueError(
                        f"HF token lacks Inference API permission. Create a token with read access. {msg}"
                    )
                raise RuntimeError(f"HF API error: {msg}")

            result = response.json()
            if isinstance(result, list) and result and isinstance(result[0], list):
                batch_emb = result
            else:
                batch_emb = [
                    item if isinstance(item, list) else [item]
                    for item in (result if isinstance(result, list) else [result])
                ]

            # Normalize for cosine similarity (Milvus COSINE)
            for vec in batch_emb:
                norm = sum(x * x for x in vec) ** 0.5
                if norm == 0:
                    norm = 1.0
                all_embeddings.append([x / norm for x in vec])

    return all_embeddings


def get_milvus_client() -> Any:
    """
    Connect to Milvus Cloud and return a client. Creates collection "documents"
    if it does not exist (dim 384 for all-MiniLM-L6-v2).
    """
    if not MILVUS_URI or not MILVUS_TOKEN:
        raise ValueError("MILVUS_URI and MILVUS_TOKEN must be set in .env")

    from pymilvus import MilvusClient

    client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)
    logger.info("Milvus connection established")

    if not client.has_collection(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            dimension=VECTOR_DIM,
            primary_field_name="id",
            vector_field_name="vector",
            metric_type="COSINE",
            auto_id=True,
        )
        logger.info("Collection %s created (dim=%s)", COLLECTION_NAME, VECTOR_DIM)
    return client


def store_chunks(chunks: list[dict]) -> None:
    """
    Embed each chunk with HF (all-MiniLM-L6-v2), insert into Milvus with metadata
    (text, source, chunk_id), then flush the collection.
    """
    if not chunks:
        return

    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)

    client = get_milvus_client()
    rows = []
    for c, emb in zip(chunks, embeddings):
        meta = c.get("metadata", {})
        rows.append({
            "vector": emb,
            "text": c["text"],
            "source": meta.get("source", ""),
            "chunk_id": meta.get("chunk_id", 0),
        })

    client.insert(collection_name=COLLECTION_NAME, data=rows)
    client.flush(collection_name=COLLECTION_NAME)
    logger.info("Embedded and stored %d chunks", len(chunks))
