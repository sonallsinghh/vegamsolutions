"""
Agent tools: definitions and execution for tool-calling (agentic) mode.

Tools: search_documents, list_sources, get_chunk, system_stats, current_date, calculator, web_search (optional).
"""

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any

from app.services.retrieval_service import retrieve_context
from app.services.vector_store import get_chunk_by_id, get_collection_stats, list_sources

logger = logging.getLogger(__name__)

# OpenAI function-calling format: list of tool definitions
AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "Search the knowledge base using semantic retrieval. Use this to find relevant passages, recipes, policies, or any content from uploaded documents. Returns chunks with id, text, and source.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (keywords or natural language question)",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_sources",
            "description": "List all document source names in the knowledge base (e.g. file names). Use to see what documents are available before or after searching.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_chunk",
            "description": "Fetch a single chunk by its id. Use after search_documents when you need the full text of a specific chunk (id is returned in search results).",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "integer",
                        "description": "Chunk id (Milvus primary key from search_documents results)",
                    }
                },
                "required": ["id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "system_stats",
            "description": "Get knowledge base statistics: total chunks, source count, and list of sources. Use for system observability or to check if the KB is populated.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "current_date",
            "description": "Get the current date and time (UTC). Use when the user asks about today, now, or time-sensitive information.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a simple math expression. Use for numeric calculations (e.g. 2+3*4, 100/5). Only numbers and + - * / ( ) . allowed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression to evaluate (e.g. 2+3*4)",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current or external information. Use when the answer is not in the knowledge base or the user asks about recent events, weather, or general knowledge.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for the web",
                    }
                },
                "required": ["query"],
            },
        },
    },
]


def _safe_calculator(expression: str) -> str:
    """Evaluate a safe math expression (numbers and + - * / ( ) . only)."""
    expr = (expression or "").strip()
    if not expr:
        return "Error: empty expression"
    if not re.match(r"^[\d\s+\-*/().]+$", expr):
        return "Error: only numbers and + - * / ( ) . allowed"
    try:
        result = eval(expr)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def _web_search_impl(query: str) -> str:
    """Run web search if duckduckgo-search is available."""
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        logger.info("[tools] web_search: duckduckgo-search not installed; pip install duckduckgo-search")
        return "Web search is not available (install duckduckgo-search)."
    q = (query or "").strip()
    if not q:
        return "Error: empty query"
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(q, max_results=5))
        if not results:
            return "No results found."
        lines = []
        for i, r in enumerate(results[:5], 1):
            title = (r.get("title") or "").strip()
            body = (r.get("body") or "").strip()
            href = (r.get("href") or "").strip()
            lines.append(f"{i}. {title}\n{body}\nURL: {href}")
        return "\n\n".join(lines)
    except Exception as e:
        logger.warning("[tools] web_search failed: %s", e)
        return f"Web search failed: {e}"


def execute_tool(name: str, arguments: dict[str, Any]) -> str:
    """
    Execute a tool by name with the given arguments. Returns a string result for the LLM.
    """
    args = arguments or {}
    logger.info("[tools] execute_tool name=%r arguments=%r", name, args)

    if name == "search_documents":
        query = (args.get("query") or "").strip() or ""
        if not query:
            return "Error: query is required."
        chunks = retrieve_context(query)
        if not chunks:
            return "No matching chunks found."
        results = []
        for c in chunks[:8]:
            text = (c.get("text") or "")[:800]
            source = (c.get("metadata") or {}).get("source", "")
            pk = c.get("id")
            results.append(f"[id={pk} source={source}]\n{text}")
        return "\n\n---\n\n".join(results)

    if name == "list_sources":
        sources = list_sources()
        if not sources:
            return "Knowledge base is empty (no sources)."
        return "Sources in knowledge base:\n" + "\n".join(f"- {s}" for s in sources)

    if name == "get_chunk":
        chunk_id = args.get("id")
        if chunk_id is None:
            return "Error: id is required."
        chunk = get_chunk_by_id(chunk_id)
        if chunk is None:
            return f"No chunk found with id={chunk_id}."
        return json.dumps(chunk, default=str)

    if name == "system_stats":
        stats = get_collection_stats()
        return json.dumps(stats, indent=2)

    if name == "current_date":
        now = datetime.now(timezone.utc)
        return now.strftime("%Y-%m-%d %H:%M:%S UTC")

    if name == "calculator":
        expr = args.get("expression") or ""
        return _safe_calculator(expr)

    if name == "web_search":
        query = (args.get("query") or "").strip()
        return _web_search_impl(query)

    return f"Unknown tool: {name}"
