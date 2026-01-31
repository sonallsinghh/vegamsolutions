"""
Lightweight SQLite DB for storing file paths of uploaded files.

Creates data/uploads.db (relative to project root). Table: uploads (id, path, created_at).
"""

import sqlite3
import logging
from pathlib import Path
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Project root (VegamSolutions)
_ROOT = Path(__file__).resolve().parent.parent.parent
_DB_PATH = _ROOT / "data" / "uploads.db"
_TABLE = "uploads"


def _get_conn() -> sqlite3.Connection:
    data_dir = _DB_PATH.parent
    data_dir.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(str(_DB_PATH))


def init_db() -> None:
    """Create the uploads table if it does not exist."""
    conn = _get_conn()
    try:
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {_TABLE} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def add_path(path: str) -> None:
    """Insert a file path. Path is relative (e.g. data/uploads/file.txt)."""
    if not path or not str(path).strip():
        return
    path = str(path).strip()
    init_db()
    conn = _get_conn()
    try:
        conn.execute(
            f"INSERT INTO {_TABLE} (path, created_at) VALUES (?, ?)",
            (path, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        logger.info("[upload_db] added path=%s", path)
    finally:
        conn.close()


def get_all_paths() -> list[str]:
    """Return all stored file paths, oldest first."""
    init_db()
    conn = _get_conn()
    try:
        cur = conn.execute(f"SELECT path FROM {_TABLE} ORDER BY id ASC")
        return [row[0] for row in cur.fetchall()]
    finally:
        conn.close()


def clear_all() -> None:
    """Delete all rows. Call when clearing the knowledge base."""
    init_db()
    conn = _get_conn()
    try:
        conn.execute(f"DELETE FROM {_TABLE}")
        conn.commit()
        logger.info("[upload_db] cleared all paths")
    finally:
        conn.close()
