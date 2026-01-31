"""
Minimal MCP-style tool server: exposes retrieval as a standardized tool interface.
This endpoint acts as an MCP tool server, allowing external agents to call
retrieval through a standardized interface.
"""

import logging
from fastapi import APIRouter
from pydantic import BaseModel

from app.services.retrieval_service import retrieve_context

logger = logging.getLogger(__name__)

# MCP tool schema for discovery / documentation
tools = [
    {
        "name": "search_documents",
        "description": "Search documents using semantic retrieval",
        "input_schema": {
            "query": "string",
        },
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
def mcp_search_documents(body: SearchDocumentsRequest) -> dict[str, list[dict[str, str]]]:
    """
    This endpoint acts as an MCP tool server,
    allowing external agents to call retrieval
    through a standardized interface.
    """
    logger.info("MCP tool called: search_documents")
    query = (body.query or "").strip()
    if not query:
        return {"results": []}
    chunks = retrieve_context(query)
    results = [
        {
            "text": c.get("text", ""),
            "source": (c.get("metadata") or {}).get("source", ""),
        }
        for c in chunks
    ]
    return {"results": results}
