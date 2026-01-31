"""
Application errors for clean API error handling.

Use ServiceUnavailableError when a dependency (vector store, embeddings, LLM)
is misconfigured or unreachable so the API can return 503 with a user-facing message.
"""


class ServiceUnavailableError(Exception):
    """Raised when a required service (e.g. vector store, embeddings API) is unavailable or misconfigured."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)
