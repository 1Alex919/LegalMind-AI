"""Tests for document ingestion pipeline."""

import pytest

from src.ingestion.chunking import Chunk, chunk_document
from src.ingestion.loaders import DocumentPage, LoadedDocument


@pytest.fixture
def mock_document() -> LoadedDocument:
    text = (
        "This Non-Disclosure Agreement (NDA) is entered into by and between "
        "Party A and Party B effective January 1, 2024.\n\n"
        "Section 1: Confidential Information\n"
        "Confidential information includes all technical, business, and financial "
        "information disclosed by either party during the term of this agreement. "
        "This includes but is not limited to trade secrets, customer lists, "
        "product roadmaps, and pricing strategies.\n\n"
        "Section 2: Obligations\n"
        "The receiving party shall not disclose confidential information to any "
        "third party without prior written consent. The receiving party shall use "
        "the same degree of care to protect the disclosing party's confidential "
        "information as it uses to protect its own.\n\n"
        "Section 3: Term and Termination\n"
        "This agreement shall remain in effect for a period of 2 years from the "
        "effective date. Either party may terminate this agreement with 30 days "
        "written notice to the other party."
    )
    return LoadedDocument(
        pages=[DocumentPage(text=text, page_number=1, metadata={"source": "test.pdf", "page": 1})],
        filename="test.pdf",
        file_type="pdf",
        total_pages=1,
    )


def test_chunk_document_creates_parent_and_child(mock_document: LoadedDocument) -> None:
    chunked = chunk_document(mock_document)
    assert len(chunked.parent_chunks) > 0
    assert len(chunked.child_chunks) > 0
    assert chunked.filename == "test.pdf"


def test_child_chunks_reference_parent(mock_document: LoadedDocument) -> None:
    chunked = chunk_document(mock_document)
    for child in chunked.child_chunks:
        assert child.parent_id is not None
        parent_ids = {p.chunk_id for p in chunked.parent_chunks}
        assert child.parent_id in parent_ids


def test_chunk_metadata_includes_source(mock_document: LoadedDocument) -> None:
    chunked = chunk_document(mock_document)
    for chunk in chunked.child_chunks:
        assert "filename" in chunk.metadata
        assert chunk.metadata["filename"] == "test.pdf"
        assert chunk.metadata["chunk_type"] == "child"


def test_loaded_document_full_text(mock_document: LoadedDocument) -> None:
    assert "Non-Disclosure Agreement" in mock_document.full_text
    assert "termination" in mock_document.full_text.lower()
