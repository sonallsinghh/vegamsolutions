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
# Use router (api-inference.huggingface.co returns 410 Gone)
HF_RERANK_URL = f"https://router.huggingface.co/hf-inference/models/{HF_RERANK_MODEL}"
RERANK_TOP_K = 8
SEARCH_TOP_K = 100
API_TIMEOUT = 60.0


def search_milvus(query: str, top_k: int = SEARCH_TOP_K) -> list[dict]:
    """
    Embed query, search Milvus, return structured candidates with text, score, metadata.
    """
    logger.info("[retrieval:search_milvus] IN  query=%r top_k=%d", query, top_k)
    if not query or not query.strip():
        logger.info("[retrieval:search_milvus] OUT empty query, returning []")
        return []

    query_vec = embed_texts([query.strip()])
    if not query_vec:
        logger.warning("[retrieval:search_milvus] embed_texts returned empty")
        return []

    client = get_milvus_client()
    results = client.search(
        collection_name=COLLECTION_NAME,
        data=query_vec,
        limit=top_k,
        output_fields=["id", "text", "source", "chunk_id"],
    )

    # results: list of list of hits (one list per query vector)
    hits = results[0] if results else []
    candidates = []
    for h in hits:
        # Milvus returns dict with "distance", "id", and optionally "entity" (output_fields)
        score = float(h.get("distance", h.get("score", 0.0)))
        entity = h.get("entity") if "entity" in h else h
        e = entity or h
        text = e.get("text", "")
        candidates.append({
            "id": e.get("id", h.get("id")),
            "text": text,
            "score": score,
            "metadata": {
                "source": e.get("source", ""),
                "chunk_id": e.get("chunk_id", 0),
            },
        })
    logger.info("[retrieval:search_milvus] OUT candidates=%d first_sources=%s first_scores=%s",
                len(candidates),
                [c.get("metadata", {}).get("source") for c in candidates[:5]],
                [round(c.get("score", 0), 4) for c in candidates[:5]])
    for i, c in enumerate(candidates[:8]):
        logger.info("[retrieval:search_milvus] candidate_%d source=%s score=%.4f text_preview=%r",
                    i + 1, c.get("metadata", {}).get("source"), c.get("score", 0), (c.get("text") or "")[:200])
    return candidates


def rerank_with_hf(query: str, results: list[dict], top_k: int = RERANK_TOP_K) -> list[dict]:
    """
    Rerank candidates using Hugging Face Inference API (BAAI/bge-reranker-base).

    Reranking improves precision after high-recall vector search.
    Returns top_k chunks sorted by relevance score.
    """
    if not results or not query or not HF_API_KEY:
        return results[:top_k] if results else []

    # Router reranker expects list of {"text": query, "text_pair": document}
    inputs = [{"text": query, "text_pair": r["text"]} for r in results]
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {"inputs": inputs, "options": {"wait_for_model": True}}

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

    # API returns list of scores (same order as inputs). Router can return [[s1,s2,...]] (one list of scores).
    def to_score(item, i: int) -> float:
        if isinstance(item, (int, float)):
            return float(item)
        if isinstance(item, list) and item:
            return float(item[0]) if isinstance(item[0], (int, float)) else 0.0
        if isinstance(item, dict):
            return float(item.get("score", 0))
        return 0.0

    scores_list: list = []
    if isinstance(data, list):
        if not data:
            return results[:top_k]
        # Router sometimes returns [[s1, s2, ...]] — one element that is the full list of scores
        if len(data) == 1 and isinstance(data[0], list):
            scores_list = data[0]
        elif isinstance(data[0], dict) and "score" in (data[0] or {}):
            scored = [(i, to_score(item, i)) for i, item in enumerate(data)]
            scores_list = None
        else:
            scores_list = list(data)
    elif isinstance(data, dict) and "scores" in data:
        scores_list = data["scores"]
    else:
        return results[:top_k]

    if scores_list is not None:
        scored = [(i, to_score(s, i)) for i, s in enumerate(scores_list)]

    scored.sort(key=lambda x: -x[1])
    top_indices = [idx for idx, _ in scored[:top_k]]
    reranked = [results[i] for i in top_indices if i < len(results)]
    logger.info("[retrieval:rerank_with_hf] OUT reranked=%d sources=%s",
                len(reranked), [r.get("metadata", {}).get("source") for r in reranked])
    for i, r in enumerate(reranked):
        logger.info("[retrieval:rerank_with_hf] rerank_%d source=%s text_preview=%r",
                    i + 1, r.get("metadata", {}).get("source"), (r.get("text") or "")[:200])
    return reranked


def _boost_by_keywords(query: str, candidates: list[dict]) -> list[dict]:
    """
    Reorder candidates so chunks containing query words (e.g. 'maternity', 'leave')
    come first. Ensures exact phrases in the doc are not missed when vector rank is low.
    """
    if not candidates or not query or not query.strip():
        return candidates
    words = [w for w in query.lower().split() if len(w) >= 2]
    if not words:
        return candidates

    def keyword_score(c: dict) -> int:
        text = (c.get("text") or "").lower()
        return sum(1 for w in words if w in text)

    # Sort: more query words in chunk first, then by vector score (higher is better)
    scored = [(c, keyword_score(c)) for c in candidates]
    scored.sort(key=lambda x: (-x[1], -x[0].get("score", 0)))
    return [c for c, _ in scored]


def retrieve_context(query: str) -> list[dict]:
    """
    Pipeline: semantic search (Milvus) → keyword boost → HF rerank → top chunks.
    """
    logger.info("[retrieval:retrieve_context] IN  query=%r", query)
    candidates = search_milvus(query, top_k=SEARCH_TOP_K)
    logger.info("[retrieval:retrieve_context] after search candidates=%d", len(candidates))
    candidates = _boost_by_keywords(query, candidates)
    logger.info("[retrieval:retrieve_context] after keyword_boost candidates=%d first_sources=%s",
               len(candidates), [c.get("metadata", {}).get("source") for c in candidates[:3]])
    reranked = rerank_with_hf(query, candidates, top_k=RERANK_TOP_K)
    logger.info("[retrieval:retrieve_context] OUT retrieved %d → reranked to %d", len(candidates), len(reranked))
    return reranked
