"""
Vector store client: Milvus Cloud connection, embeddings (HF Inference API), and chunk storage.

Responsibility: Connect to Milvus, embed texts via all-MiniLM-L6-v2, store chunks with metadata.
"""

import logging
from typing import Any

import httpx

from app.core.config import (
    COLLECTION_NAME,
    EMBED_API_TIMEOUT,
    EMBED_BATCH_SIZE,
    HF_API_KEY,
    HF_EMBED_MODEL,
    MILVUS_TOKEN,
    MILVUS_URI,
    VECTOR_DIM,
)

logger = logging.getLogger(__name__)

HF_API_URL_ROUTER = (
    "https://router.huggingface.co/hf-inference/models/"
    f"{HF_EMBED_MODEL}/pipeline/feature-extraction"
)
HF_API_URL_STANDARD = f"https://api-inference.huggingface.co/models/{HF_EMBED_MODEL}"


def embed_texts(
    texts: list[str], batch_size: int | None = None
) -> list[list[float]]:
    """
    Batch embed texts using Hugging Face Inference API (all-MiniLM-L6-v2).

    Batch embeddings reduce latency and improve throughput.
    Returns list of 384-dim vectors (normalized for cosine similarity).
    """
    batch_size = batch_size if batch_size is not None else EMBED_BATCH_SIZE
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

    with httpx.Client(timeout=EMBED_API_TIMEOUT) as client:
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


def list_sources(limit: int = 16_384) -> list[str]:
    """
    Return distinct document source names in the collection (document introspection).
    Used by MCP list_sources tool.
    """
    client = get_milvus_client()
    if not client.has_collection(COLLECTION_NAME):
        return []
    results = client.query(
        collection_name=COLLECTION_NAME,
        filter="",
        limit=limit,
        output_fields=["source"],
    )
    sources = sorted({(r.get("source") or "").strip() for r in results if (r.get("source") or "").strip()})
    return sources


def clear_knowledge_base() -> None:
    """
    Remove all data from the knowledge base by dropping the Milvus collection.
    The collection will be recreated empty on next insert or get_milvus_client() call.
    """
    client = get_milvus_client()
    if client.has_collection(COLLECTION_NAME):
        client.drop_collection(collection_name=COLLECTION_NAME)
        logger.info("Knowledge base cleared: collection %s dropped", COLLECTION_NAME)


def get_chunk_by_id(chunk_id: int | str) -> dict | None:
    """
    Fetch a single chunk by its Milvus primary key (id). Returns dict with id, text, source, chunk_id.
    Used by MCP get_chunk tool (metadata awareness).
    """
    client = get_milvus_client()
    if not client.has_collection(COLLECTION_NAME):
        return None
    try:
        results = client.get(
            collection_name=COLLECTION_NAME,
            ids=[chunk_id],
            output_fields=["text", "source", "chunk_id"],
        )
    except Exception:
        return None
    if not results:
        return None
    r = results[0]
    return {
        "id": r.get("id", chunk_id),
        "text": r.get("text", ""),
        "source": r.get("source", ""),
        "chunk_id": r.get("chunk_id", 0),
    }


def get_collection_stats() -> dict:
    """
    Return knowledge-base stats: total chunks, source count, sources list, collection name.
    Used by MCP system_stats tool (system observability).
    """
    client = get_milvus_client()
    if not client.has_collection(COLLECTION_NAME):
        return {
            "collection_name": COLLECTION_NAME,
            "total_chunks": 0,
            "source_count": 0,
            "sources": [],
        }
    total = client.num_entities(collection_name=COLLECTION_NAME)
    sources = list_sources(limit=16_384)
    return {
        "collection_name": COLLECTION_NAME,
        "total_chunks": total,
        "source_count": len(sources),
        "sources": sources,
    }
