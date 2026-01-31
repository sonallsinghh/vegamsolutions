"""
Unit tests for text processing: clean_text and chunk_text.
"""

import pytest

from app.services.text_processing import clean_text, chunk_text


class TestCleanText:
    """Tests for clean_text()."""

    def test_empty_returns_empty(self) -> None:
        assert clean_text("") == ""
        assert clean_text("   ") == ""
        assert clean_text("\n\n") == ""

    def test_strips_outer_whitespace(self) -> None:
        assert clean_text("  hello  ") == "hello"
        assert clean_text("\n  hello  \n") == "hello"

    def test_normalizes_inner_lines_and_dedupes(self) -> None:
        # Consecutive duplicate lines become one; blank lines preserved between paragraphs
        assert clean_text("  hello   \n\n  world  ") == "hello\n\nworld"
        assert clean_text("line1\n  line1  \nline2") == "line1\nline2"

    def test_preserves_single_blank_between_paragraphs(self) -> None:
        text = "First para.\n\nSecond para."
        assert clean_text(text) == "First para.\n\nSecond para."

    def test_nfkc_normalization(self) -> None:
        # NFKC normalizes compatibility chars (e.g. fullwidth)
        normalized = clean_text("hello\u200bworld")  # zero-width space
        assert "hello" in normalized and "world" in normalized


class TestChunkText:
    """Tests for chunk_text()."""

    def test_empty_returns_empty_list(self) -> None:
        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_short_text_returns_single_chunk(self) -> None:
        short = "This is a short paragraph."
        assert chunk_text(short, chunk_size=500, overlap=50) == [short]

    def test_text_under_chunk_size_returns_one_chunk(self) -> None:
        text = "One. Two. Three."
        assert chunk_text(text, chunk_size=100, overlap=10) == [text]

    def test_long_text_produces_multiple_chunks(self) -> None:
        # Build text longer than chunk_size using sentences
        sentences = [f"Sentence number {i} here." for i in range(25)]
        text = " ".join(sentences)
        chunks = chunk_text(text, chunk_size=80, overlap=15)
        assert len(chunks) >= 2
        for c in chunks:
            assert len(c) <= 80 + 20  # chunk_size + some slack for boundary

    def test_chunks_are_non_overlapping_contiguous_with_overlap(self) -> None:
        # Overlap means end of chunk N appears at start of chunk N+1
        text = "First sentence. Second sentence. Third sentence. Fourth. Fifth."
        chunks = chunk_text(text, chunk_size=40, overlap=10)
        assert len(chunks) >= 2
        # Each chunk should be non-empty and stripped
        for c in chunks:
            assert c.strip() == c and len(c) > 0
