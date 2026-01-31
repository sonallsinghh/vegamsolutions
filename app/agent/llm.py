"""
Agent LLM: OpenAI (primary) or Hugging Face (fallback).
When OPENAI_API_KEY is set, uses OpenAI chat completions; otherwise uses HF router.
"""

import json
import logging
from typing import Any

import httpx

from app.core.config import (
    HF_API_KEY,
    HF_LLM_MODEL,
    OPENAI_API_KEY,
    OPENAI_LLM_MODEL,
)

logger = logging.getLogger(__name__)
API_TIMEOUT = 60.0

# Hugging Face router (fallback)
HF_CHAT_URL = "https://router.huggingface.co/v1/chat/completions"


def _call_openai(prompt: str, max_new_tokens: int) -> str:
    """Call OpenAI chat completions. Returns generated text."""
    try:
        from openai import OpenAI
    except ImportError:
        logger.warning("[llm] openai package not installed; pip install openai")
        return ""
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=OPENAI_LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_new_tokens,
    )
    msg = response.choices[0].message if response.choices else None
    if not msg or not getattr(msg, "content", None):
        return ""
    out = (msg.content or "").strip()
    logger.info("[llm:openai] OUT response_len=%d", len(out))
    logger.info("[llm:openai] OUT response_full=%r", out)
    return out


def _call_hf(prompt: str, max_new_tokens: int) -> str:
    """Call Hugging Face router chat completions. Returns generated text."""
    if not HF_API_KEY:
        logger.warning("[llm:hf] no HF_API_KEY")
        return ""
    headers = {"Authorization": f"Bearer {HF_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": HF_LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_new_tokens,
    }
    try:
        with httpx.Client(timeout=API_TIMEOUT) as client:
            response = client.post(HF_CHAT_URL, json=payload, headers=headers)
        if response.status_code != 200:
            logger.warning("[llm:hf] HF LLM error %s: %s", response.status_code, response.text[:200])
            return ""
        data = response.json()
    except Exception as e:
        logger.warning("[llm:hf] request failed: %s", e)
        return ""
    choices = data.get("choices") or []
    if choices and isinstance(choices[0], dict):
        msg = choices[0].get("message") or {}
        out = (msg.get("content") or "").strip()
        logger.info("[llm:hf] OUT response_len=%d", len(out))
        logger.info("[llm:hf] OUT response_full=%r", out)
        return out
    return ""


def hf_llm(prompt: str, max_new_tokens: int = 256) -> str:
    """
    Call LLM for text generation. Uses OpenAI when OPENAI_API_KEY is set, else Hugging Face.
    If OpenAI is chosen but fails (e.g. package not installed) or returns empty, falls back to HF.
    Returns generated text.
    """
    logger.info("[llm] IN  prompt_len=%d max_new_tokens=%d", len(prompt), max_new_tokens)
    logger.debug("[llm] prompt_sample=%r", prompt[:500] if len(prompt) > 500 else prompt)
    if OPENAI_API_KEY:
        out = _call_openai(prompt, max_new_tokens)
        if out:
            return out
        logger.info("[llm] OpenAI returned empty; falling back to Hugging Face")
    return _call_hf(prompt, max_new_tokens)


def chat_with_tools(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    max_tokens: int = 512,
) -> tuple[str | None, list[dict[str, Any]] | None]:
    """
    Call OpenAI chat with tools. Used for agentic (tool-calling) mode.
    Returns (content, tool_calls). If tool_calls is non-empty, caller should execute
    them and call again with tool results; if content is set and no tool_calls, that's the final answer.
    Requires OPENAI_API_KEY and openai package; returns (None, None) otherwise.
    """
    if not OPENAI_API_KEY:
        logger.warning("[llm] chat_with_tools requires OPENAI_API_KEY")
        return None, None
    try:
        from openai import OpenAI
    except ImportError:
        logger.warning("[llm] chat_with_tools requires openai package")
        return None, None
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=OPENAI_LLM_MODEL,
        messages=messages,
        tools=tools,
        max_tokens=max_tokens,
    )
    msg = response.choices[0].message if response.choices else None
    if not msg:
        return None, None
    content = (getattr(msg, "content", None) or "").strip() or None
    raw_tool_calls = getattr(msg, "tool_calls", None) or []
    tool_calls = []
    for tc in raw_tool_calls:
        fid = getattr(tc, "id", None) or ""
        fn = getattr(tc, "function", None)
        if not fn:
            continue
        fname = getattr(fn, "name", None) or ""
        fargs = getattr(fn, "arguments", None) or "{}"
        try:
            args = json.loads(fargs) if isinstance(fargs, str) else fargs
        except json.JSONDecodeError:
            args = {}
        tool_calls.append({"id": fid, "name": fname, "arguments": args})
    if tool_calls:
        logger.info("[llm:chat_with_tools] OUT tool_calls=%s", [t["name"] for t in tool_calls])
    if content:
        logger.info("[llm:chat_with_tools] OUT content_len=%d", len(content))
    return content, tool_calls if tool_calls else None


def chat_with_tools_stream(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    max_tokens: int = 512,
):
    """
    Call OpenAI chat with tools and stream the response. Yields:
    - ('content_delta', str) for each token of the final answer;
    - ('content_done',) when the answer is complete (no tool_calls);
    - ('tool_calls', list[dict], content_str) when the model called tools (content_str may be empty).
    Requires OPENAI_API_KEY and openai package.
    """
    if not OPENAI_API_KEY:
        logger.warning("[llm] chat_with_tools_stream requires OPENAI_API_KEY")
        return
    try:
        from openai import OpenAI
    except ImportError:
        logger.warning("[llm] chat_with_tools_stream requires openai package")
        return
    client = OpenAI(api_key=OPENAI_API_KEY)
    stream = client.chat.completions.create(
        model=OPENAI_LLM_MODEL,
        messages=messages,
        tools=tools,
        max_tokens=max_tokens,
        stream=True,
    )
    content_parts: list[str] = []
    tool_calls_accum: dict[int, dict[str, Any]] = {}
    for chunk in stream:
        if not chunk.choices:
            continue
        d = chunk.choices[0].delta
        if getattr(d, "content", None) and d.content:
            content_parts.append(d.content)
            yield ("content_delta", d.content)
        if getattr(d, "tool_calls", None) and d.tool_calls:
            for tc in d.tool_calls:
                idx = getattr(tc, "index", 0)
                if idx not in tool_calls_accum:
                    tool_calls_accum[idx] = {"id": "", "name": "", "arguments": ""}
                if getattr(tc, "id", None):
                    tool_calls_accum[idx]["id"] = tc.id
                if getattr(tc, "function", None) and tc.function:
                    if getattr(tc.function, "name", None):
                        tool_calls_accum[idx]["name"] = tc.function.name
                    if getattr(tc.function, "arguments", None) and tc.function.arguments:
                        tool_calls_accum[idx]["arguments"] += tc.function.arguments
    full_content = "".join(content_parts)
    if tool_calls_accum:
        sorted_tcs = [tool_calls_accum[i] for i in sorted(tool_calls_accum)]
        tool_calls_list = []
        for t in sorted_tcs:
            try:
                args = json.loads(t["arguments"]) if t["arguments"] else {}
            except json.JSONDecodeError:
                args = {}
            tool_calls_list.append({"id": t["id"], "name": t["name"], "arguments": args})
        logger.info("[llm:chat_with_tools_stream] OUT tool_calls=%s", [x["name"] for x in tool_calls_list])
        yield ("tool_calls", tool_calls_list, full_content)
    else:
        logger.info("[llm:chat_with_tools_stream] OUT content_done len=%d", len(full_content))
        yield ("content_done",)
