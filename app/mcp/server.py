"""
Minimal MCP-style tool server: exposes retrieval and knowledge-base introspection
as a standardized tool interface. Document introspection, metadata awareness,
and system observability for RAG agents.
"""

import logging
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from app.services.retrieval_service import retrieve_context
from app.services.vector_store import get_collection_stats, get_chunk_by_id, list_sources

logger = logging.getLogger(__name__)

# MCP tool schema for discovery / documentation
tools = [
    {
        "name": "search_documents",
        "description": "Search documents using semantic retrieval",
        "input_schema": {"query": "string"},
    },
    {
        "name": "list_sources",
        "description": "List document source names in the knowledge base (document introspection)",
        "input_schema": {},
    },
    {
        "name": "get_chunk",
        "description": "Fetch a chunk by its Milvus entity id (metadata awareness)",
        "input_schema": {"id": "integer or string (Milvus primary key)"},
    },
    {
        "name": "system_stats",
        "description": "Knowledge base status: total chunks, source count, sources list (system observability)",
        "input_schema": {},
    },
]

mcp_router = APIRouter(tags=["mcp"])


class SearchDocumentsRequest(BaseModel):
    """Request body for MCP tool search_documents."""
    query: str = ""


@mcp_router.post(
    "/tools/search_documents",
    summary="MCP tool: search_documents",
    description="This endpoint acts as an MCP tool server, allowing external agents to call retrieval through a standardized interface.",
)
def mcp_search_documents(body: SearchDocumentsRequest) -> dict[str, list[dict[str, Any]]]:
    """
    This endpoint acts as an MCP tool server,
    allowing external agents to call retrieval
    through a standardized interface.
    Each result includes id (Milvus primary key) so the agent can call get_chunk(id).
    """
    logger.info("MCP tool called: search_documents")
    query = (body.query or "").strip()
    if not query:
        return {"results": []}
    chunks = retrieve_context(query)
    results = [
        {
            "id": c.get("id"),
            "text": c.get("text", ""),
            "source": (c.get("metadata") or {}).get("source", ""),
        }
        for c in chunks
    ]
    return {"results": results}


# --- list_sources ---

@mcp_router.post(
    "/tools/list_sources",
    summary="MCP tool: list_sources",
    description="List document source names in the knowledge base (document introspection).",
)
def mcp_list_sources() -> dict[str, list[str]]:
    """List what documents exist in the knowledge base."""
    logger.info("MCP tool called: list_sources")
    sources = list_sources()
    return {"sources": sources}


# --- get_chunk ---

class GetChunkRequest(BaseModel):
    """Request body for MCP tool get_chunk."""
    id: int | str


@mcp_router.post(
    "/tools/get_chunk",
    summary="MCP tool: get_chunk",
    description="Fetch a chunk by its Milvus entity id (metadata awareness).",
)
def mcp_get_chunk(body: GetChunkRequest) -> dict[str, Any]:
    """Fetch chunk by id. Returns chunk with id, text, source, chunk_id or null if not found."""
    logger.info("MCP tool called: get_chunk")
    chunk = get_chunk_by_id(body.id)
    if chunk is None:
        return {"chunk": None}
    return {"chunk": chunk}


# --- system_stats ---

@mcp_router.post(
    "/tools/system_stats",
    summary="MCP tool: system_stats",
    description="Knowledge base status: total chunks, source count, sources list (system observability).",
)
def mcp_system_stats() -> dict[str, Any]:
    """Return knowledge base status for system observability."""
    logger.info("MCP tool called: system_stats")
    stats = get_collection_stats()
    return stats
