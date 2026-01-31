"""
Integration tests for MCP tool endpoints.

Uses mocks for retrieval/vector_store so tests do not require Milvus or HF API.
"""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_mcp_search_documents_returns_results(client: TestClient) -> None:
    """POST /mcp/tools/search_documents returns 200 and { results: [{ id, text, source }] }."""
    fake_chunks = [
        {"id": 101, "text": "Fake chunk one.", "metadata": {"source": "doc1.txt", "chunk_id": 0}},
        {"id": 102, "text": "Fake chunk two.", "metadata": {"source": "doc2.txt", "chunk_id": 1}},
    ]
    with patch("app.mcp.server.retrieve_context", return_value=fake_chunks):
        response = client.post(
            "/mcp/tools/search_documents",
            json={"query": "test question"},
        )
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert data["results"] == [
        {"id": 101, "text": "Fake chunk one.", "source": "doc1.txt"},
        {"id": 102, "text": "Fake chunk two.", "source": "doc2.txt"},
    ]


def test_mcp_search_documents_empty_query_returns_empty_results(client: TestClient) -> None:
    """POST with empty query returns 200 and empty results (retrieve_context not called)."""
    with patch("app.mcp.server.retrieve_context") as mock_retrieve:
        response = client.post(
            "/mcp/tools/search_documents",
            json={"query": ""},
        )
    assert response.status_code == 200
    assert response.json() == {"results": []}
    mock_retrieve.assert_not_called()


def test_mcp_search_documents_missing_body_returns_422(client: TestClient) -> None:
    """POST without body or invalid JSON returns 422."""
    response = client.post("/mcp/tools/search_documents")
    assert response.status_code == 422


# --- list_sources ---

def test_mcp_list_sources_returns_sources(client: TestClient) -> None:
    """POST /mcp/tools/list_sources returns 200 and { sources: [...] }."""
    with patch("app.mcp.server.list_sources", return_value=["doc1.txt", "doc2.pdf"]):
        response = client.post("/mcp/tools/list_sources", json={})
    assert response.status_code == 200
    assert response.json() == {"sources": ["doc1.txt", "doc2.pdf"]}


def test_mcp_list_sources_empty(client: TestClient) -> None:
    """POST /mcp/tools/list_sources with empty KB returns empty list."""
    with patch("app.mcp.server.list_sources", return_value=[]):
        response = client.post("/mcp/tools/list_sources", json={})
    assert response.status_code == 200
    assert response.json() == {"sources": []}


# --- get_chunk ---

def test_mcp_get_chunk_returns_chunk(client: TestClient) -> None:
    """POST /mcp/tools/get_chunk returns 200 and { chunk: { id, text, source, chunk_id } }."""
    fake = {"id": 42, "text": "Chunk content.", "source": "doc.txt", "chunk_id": 1}
    with patch("app.mcp.server.get_chunk_by_id", return_value=fake):
        response = client.post("/mcp/tools/get_chunk", json={"id": 42})
    assert response.status_code == 200
    assert response.json() == {"chunk": fake}


def test_mcp_get_chunk_not_found_returns_null(client: TestClient) -> None:
    """POST /mcp/tools/get_chunk when id not found returns { chunk: null }."""
    with patch("app.mcp.server.get_chunk_by_id", return_value=None):
        response = client.post("/mcp/tools/get_chunk", json={"id": 999})
    assert response.status_code == 200
    assert response.json() == {"chunk": None}


# --- system_stats ---

def test_mcp_system_stats_returns_stats(client: TestClient) -> None:
    """POST /mcp/tools/system_stats returns 200 and knowledge-base stats."""
    fake = {
        "collection_name": "documents",
        "total_chunks": 100,
        "source_count": 3,
        "sources": ["a.txt", "b.pdf", "c.xlsx"],
    }
    with patch("app.mcp.server.get_collection_stats", return_value=fake):
        response = client.post("/mcp/tools/system_stats", json={})
    assert response.status_code == 200
    assert response.json() == fake
