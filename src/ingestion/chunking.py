"""Smart document chunking with parent-child strategy."""

from dataclasses import dataclass, field
from uuid import uuid4

from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from config.settings import settings
from src.ingestion.loaders import LoadedDocument


@dataclass
class Chunk:
    """A chunk of text with metadata and parent reference."""

    chunk_id: str
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


@dataclass
class ChunkedDocument:
    """Document split into parent and child chunks."""

    parent_chunks: list[Chunk]
    child_chunks: list[Chunk]
    filename: str


def chunk_document(document: LoadedDocument) -> ChunkedDocument:
    """Split document into parent (large) and child (small) chunks.

    Parent chunks provide broader context for retrieval.
    Child chunks are used for precise embedding and search.
    Each child references its parent for context expansion.
    """
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE * 3,
        chunk_overlap=settings.CHUNK_OVERLAP * 2,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    parent_chunks: list[Chunk] = []
    child_chunks: list[Chunk] = []

    for page in document.pages:
        parent_texts = parent_splitter.split_text(page.text)

        for parent_text in parent_texts:
            parent_id = str(uuid4())
            parent_chunks.append(
                Chunk(
                    chunk_id=parent_id,
                    text=parent_text,
                    metadata={
                        **page.metadata,
                        "chunk_type": "parent",
                        "filename": document.filename,
                    },
                )
            )

            child_texts = child_splitter.split_text(parent_text)
            for child_text in child_texts:
                child_chunks.append(
                    Chunk(
                        chunk_id=str(uuid4()),
                        text=child_text,
                        parent_id=parent_id,
                        metadata={
                            **page.metadata,
                            "chunk_type": "child",
                            "parent_id": parent_id,
                            "filename": document.filename,
                        },
                    )
                )

    logger.info(
        f"Chunked {document.filename}: "
        f"{len(parent_chunks)} parents, {len(child_chunks)} children"
    )

    return ChunkedDocument(
        parent_chunks=parent_chunks,
        child_chunks=child_chunks,
        filename=document.filename,
    )
