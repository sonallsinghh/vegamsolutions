"""
Retrieval: semantic search, HF rerank, and context pipeline.

Responsibility: Query Milvus, rerank with HF, return top chunks for the agent.
"""

import logging
from typing import Any

import httpx

from app.core.config import HF_API_KEY
from app.services.vector_store import embed_texts, get_milvus_client

logger = logging.getLogger(__name__)

COLLECTION_NAME = "documents"
HF_RERANK_MODEL = "BAAI/bge-reranker-base"
HF_RERANK_URL = f"https://api-inference.huggingface.co/models/{HF_RERANK_MODEL}"
RERANK_TOP_K = 5
SEARCH_TOP_K = 20
API_TIMEOUT = 60.0


def search_milvus(query: str, top_k: int = SEARCH_TOP_K) -> list[dict]:
    """
    Embed query, search Milvus, return structured candidates with text, score, metadata.
    """
    if not query or not query.strip():
        return []

    query_vec = embed_texts([query.strip()])
    if not query_vec:
        return []

    client = get_milvus_client()
    results = client.search(
        collection_name=COLLECTION_NAME,
        data=query_vec,
        limit=top_k,
        output_fields=["text", "source", "chunk_id"],
    )

    # results: list of list of hits (one list per query vector)
    hits = results[0] if results else []
    candidates = []
    for h in hits:
        # Milvus returns dict with "distance" and optionally "entity" (output_fields)
        score = float(h.get("distance", h.get("score", 0.0)))
        entity = h.get("entity") if "entity" in h else h
        text = (entity or {}).get("text", "")
        candidates.append({
            "text": text,
            "score": score,
            "metadata": {
                "source": (entity or {}).get("source", ""),
                "chunk_id": (entity or {}).get("chunk_id", 0),
            },
        })
    return candidates


def rerank_with_hf(query: str, results: list[dict], top_k: int = RERANK_TOP_K) -> list[dict]:
    """
    Rerank candidates using Hugging Face Inference API (BAAI/bge-reranker-base).

    Reranking improves precision after high-recall vector search.
    Returns top_k chunks sorted by relevance score.
    """
    if not results or not query or not HF_API_KEY:
        return results[:top_k] if results else []

    # HF reranker: inputs as list of [query, document] pairs
    pairs = [[query, r["text"]] for r in results]
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {"inputs": pairs, "options": {"wait_for_model": True}}

    try:
        with httpx.Client(timeout=API_TIMEOUT) as client:
            response = client.post(HF_RERANK_URL, json=payload, headers=headers)
        if response.status_code != 200:
            logger.warning("Reranker API error %s: %s", response.status_code, response.text[:200])
            return results[:top_k]
        data = response.json()
    except Exception as e:
        logger.warning("Reranker request failed: %s", e)
        return results[:top_k]

    # API returns list of scores (same order as pairs) or list of {index, score}
    if isinstance(data, list):
        if data and isinstance(data[0], dict):
            scored = [(item.get("index", i), float(item.get("score", 0))) for i, item in enumerate(data)]
        else:
            scored = [(i, float(s)) for i, s in enumerate(data)]
    elif isinstance(data, dict) and "scores" in data:
        scored = [(i, float(s)) for i, s in enumerate(data["scores"])]
    else:
        return results[:top_k]

    scored.sort(key=lambda x: -x[1])
    top_indices = [idx for idx, _ in scored[:top_k]]
    return [results[i] for i in top_indices if i < len(results)]


def retrieve_context(query: str) -> list[dict]:
    """
    Pipeline: semantic search (Milvus) → HF rerank → return top 5 chunks.
    """
    candidates = search_milvus(query, top_k=SEARCH_TOP_K)
    reranked = rerank_with_hf(query, candidates, top_k=RERANK_TOP_K)
    logger.info("Retrieved %d → reranked to %d", len(candidates), len(reranked))
    return reranked
