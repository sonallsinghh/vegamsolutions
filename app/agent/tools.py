"""
Agent tools: definitions and execution for tool-calling (agentic) mode.

Tools: search_documents, list_sources, get_chunk, system_stats, current_date, calculator,
get_weather (Open-Meteo API), web_search (optional).
"""

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any

import httpx

from app.services.retrieval_service import retrieve_context
from app.services.vector_store import get_chunk_by_id, get_collection_stats, list_sources

# Open-Meteo: free weather API, no key. https://open-meteo.com/en/docs
OPEN_METEO_GEOCODE = "https://geocoding-api.open-meteo.com/v1/search"
OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"
HTTP_TIMEOUT = 15.0

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
            "name": "get_weather",
            "description": "Get current weather for a city or location using Open-Meteo. Use when the user asks about weather, temperature, or conditions in a place. For small towns or regions, pass country and optionally region/state (e.g. city='Sivasagar', region='Assam', country='India') so the geocoder can find the place. Returns temperature, humidity, and conditions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City or place name (e.g. Mumbai, Sivasagar, London)",
                    },
                    "region": {
                        "type": "string",
                        "description": "Optional state/region name (e.g. Assam, California). Use for small towns so the geocoder can disambiguate.",
                    },
                    "country": {
                        "type": "string",
                        "description": "Optional country name or code (e.g. India, IN, US). Use to avoid wrong matches.",
                    }
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current or external information. Use when the answer is not in the knowledge base or the user asks about recent events or general knowledge (use get_weather for weather queries).",
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


# WMO weather codes (abbreviated) for Open-Meteo
_WMO_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


def _get_weather_impl(city: str, country: str | None = None, region: str | None = None) -> str:
    """Fetch current weather from Open-Meteo (free, no API key). Use region/country to disambiguate (e.g. Sivasagar, Assam, India)."""
    city = (city or "").strip()
    if not city:
        return "Error: city is required."
    parts = [city]
    if region and region.strip():
        parts.append(region.strip())
    if country and country.strip():
        parts.append(country.strip())
    query = ", ".join(parts) if len(parts) > 1 else city
    try:
        with httpx.Client(timeout=HTTP_TIMEOUT) as client:
            geo = client.get(OPEN_METEO_GEOCODE, params={"name": query, "count": 1})
            if geo.status_code != 200:
                return f"Weather API error: geocode returned {geo.status_code}."
            geo_data = geo.json()
            results = geo_data.get("results") or []
            if not results:
                return f"No location found for: {query}."
            loc = results[0]
            lat = loc.get("latitude")
            lon = loc.get("longitude")
            name = loc.get("name", city)
            country_code = loc.get("country_code", "")
            if lat is None or lon is None:
                return "Could not get coordinates for that location."
            forecast = client.get(
                OPEN_METEO_FORECAST,
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "current": "temperature_2m,relative_humidity_2m,weather_code",
                },
            )
            if forecast.status_code != 200:
                return f"Weather API error: forecast returned {forecast.status_code}."
            data = forecast.json()
            cur = (data.get("current") or {})
            temp = cur.get("temperature_2m")
            humidity = cur.get("relative_humidity_2m")
            code = cur.get("weather_code", 0)
            cond = _WMO_CODES.get(code, f"Weather code {code}")
            parts = [f"Location: {name} ({country_code})", f"Conditions: {cond}"]
            if temp is not None:
                parts.append(f"Temperature: {temp} °C")
            if humidity is not None:
                parts.append(f"Relative humidity: {humidity}%")
            return "\n".join(parts)
    except httpx.TimeoutException:
        return "Weather API request timed out."
    except Exception as e:
        logger.warning("[tools] get_weather failed: %s", e)
        return f"Weather lookup failed: {e}"


def _web_search_impl(query: str) -> str:
    """Run web search using ddgs (or duckduckgo_search fallback)."""
    DDGS = None
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            logger.info("[tools] web_search: install ddgs or duckduckgo-search (pip install ddgs)")
            return "Web search is not available (pip install ddgs)."
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

    if name == "get_weather":
        city = (args.get("city") or "").strip()
        region = (args.get("region") or "").strip() or None
        country = (args.get("country") or "").strip() or None
        return _get_weather_impl(city, country, region)

    if name == "web_search":
        query = (args.get("query") or "").strip()
        return _web_search_impl(query)

    return f"Unknown tool: {name}"
