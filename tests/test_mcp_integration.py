"""
Integration test for MCP tool endpoint: POST /mcp/tools/search_documents.

Uses a mock for retrieve_context so the test does not require Milvus or HF API.
"""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_mcp_search_documents_returns_results(client: TestClient) -> None:
    """POST /mcp/tools/search_documents returns 200 and { results: [{ text, source }] }."""
    fake_chunks = [
        {"text": "Fake chunk one.", "metadata": {"source": "doc1.txt", "chunk_id": 0}},
        {"text": "Fake chunk two.", "metadata": {"source": "doc2.txt", "chunk_id": 1}},
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
        {"text": "Fake chunk one.", "source": "doc1.txt"},
        {"text": "Fake chunk two.", "source": "doc2.txt"},
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
