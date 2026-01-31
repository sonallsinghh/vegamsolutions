"""
In-memory chat session store. Keyed by session_id; history is not sent from frontend.
"""

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

# session_id -> list of {"role": "user"|"assistant", "content": str}
_sessions: dict[str, list[dict[str, Any]]] = {}
_lock = threading.Lock()


def get_history(session_id: str) -> list[dict[str, Any]]:
    """Return chat history for the session (copy so caller cannot mutate store)."""
    if not session_id or not isinstance(session_id, str):
        logger.info("[session_store:get_history] IN  session_id=%r -> empty", session_id)
        return []
    with _lock:
        messages = _sessions.get(session_id) or []
        out = list(messages)
    logger.info("[session_store:get_history] IN  session_id=%s OUT messages=%d", session_id[:16], len(out))
    return out


def append_message(session_id: str, role: str, content: str) -> None:
    """Append one message to the session's history."""
    if not session_id or not isinstance(session_id, str):
        logger.info("[session_store:append_message] skip invalid session_id=%r", session_id)
        return
    with _lock:
        if session_id not in _sessions:
            _sessions[session_id] = []
        _sessions[session_id].append({"role": role, "content": content or ""})
    logger.info("[session_store:append_message] session_id=%s role=%s content_len=%d", session_id[:16], role, len(content or ""))
