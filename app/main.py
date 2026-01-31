# Run from project root: uvicorn app.main:app --reload

import logging

from fastapi import FastAPI

from app.api.routes import router

logging.basicConfig(level=logging.INFO)


app = FastAPI(title="Agentic RAG Backend")
app.include_router(router)


if __name__ == "__main__":
    print("Agentic RAG system booting...")
