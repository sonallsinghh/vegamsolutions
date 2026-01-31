#!/usr/bin/env python3
"""
Seed the uploads SQLite DB for demos or tests.

Creates data/uploads.db (if missing), ensures the uploads table exists,
and inserts seed file paths. Use --reset to clear existing rows first.

Run from project root (VegamSolutions):

    python scripts/seed_upload_db.py
    python scripts/seed_upload_db.py --reset

Seed paths match files under data/uploads/ when present; you can edit
SEED_PATHS below to add or change demo paths.
"""

import argparse
import sys
from pathlib import Path

# Project root on path so "app" resolves
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.core.upload_db import add_path, clear_all, init_db

# Paths to register as "uploaded" (relative to project root).
# Add or remove entries to match your demo files under data/uploads/.
SEED_PATHS = [
    "data/uploads/AgentX_Leave_Policy_Full.txt",
    "data/uploads/iphone_17_sample.pdf",
    "data/uploads/lpu_ranking_2010_2026.xlsx",
    "data/uploads/Sonal_AI_Engineer.pdf",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed uploads DB for demos/tests.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear all existing rows before inserting seed paths.",
    )
    args = parser.parse_args()

    init_db()
    if args.reset:
        clear_all()
        print("Cleared existing upload paths.")

    for path in SEED_PATHS:
        add_path(path)
        print(f"  added: {path}")

    print(f"Done. Seeded {len(SEED_PATHS)} paths.")


if __name__ == "__main__":
    main()
