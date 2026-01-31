"""
Application configuration (env, settings, constants).

Responsibility: Centralize config loading, environment variables, and app-wide
constants. Keeps the rest of the app decoupled from how config is sourced.
"""

import os

from dotenv import load_dotenv

load_dotenv()

# Upload storage
UPLOAD_DIR_NAME: str = "data/uploads"

# Allowed file extensions for upload and parsing
ALLOWED_EXTENSIONS: frozenset[str] = frozenset({".txt", ".pdf", ".xlsx", ".xls"})

# Chunking defaults (tuning these affects retrieval quality)
CHUNK_SIZE: int = 500
CHUNK_OVERLAP: int = 50

# Milvus Cloud (from env)
MILVUS_URI: str = os.getenv("MILVUS_URI", "").strip()
MILVUS_TOKEN: str = os.getenv("MILVUS_TOKEN", "").strip()

# Hugging Face (embeddings / inference)
HF_API_KEY: str = os.getenv("HF_API_KEY", "").strip()

# Vector collection: default embedding dim (e.g. sentence-transformers/all-MiniLM-L6-v2 = 384)
VECTOR_DIM: int = 384

# OpenAI (agent LLM). When set, the agent uses OpenAI instead of Hugging Face.
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_LLM_MODEL: str = (
    os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
)

# HF LLM for agent (fallback when OPENAI_API_KEY is not set). Router chat completions require a chat model.
HF_LLM_MODEL: str = (
    os.getenv("HF_LLM_MODEL", "meta-llama/Llama-3.2-3B-Instruct").strip()
    or "meta-llama/Llama-3.2-3B-Instruct"
)
