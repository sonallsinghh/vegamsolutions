"""Schemas for the query endpoint."""

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request body for POST /query and WebSocket /ws/query. History is stored server-side by session_id."""

    question: str = Field(..., min_length=1, description="User question for the agent.")
    session_id: str = Field(..., min_length=1, description="Session ID; chat history is stored on the server for this session.")


class QueryResponse(BaseModel):
    """Response for POST /query."""

    answer: str = Field(..., description="Final answer from the agent.")
    iterations: int = Field(0, description="Number of retrieval iterations (RAG mode). 0 when agentic.")
    chunks_used: int = Field(0, description="Number of chunks used (RAG mode). 0 when agentic.")
    tools_used: list[str] = Field(default_factory=list, description="Tools called in agentic mode (e.g. search_documents, calculator).")
