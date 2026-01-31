# Run from project root: uvicorn app.main:app --reload

import logging

from fastapi import FastAPI

from app.api.routes import router
from app.mcp.server import mcp_router

logging.basicConfig(level=logging.INFO)


app = FastAPI(title="Agentic RAG Backend")
app.include_router(router)
app.include_router(mcp_router, prefix="/mcp")


if __name__ == "__main__":
    print("Agentic RAG system booting...")
