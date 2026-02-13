"""Document ingestion pipeline."""

from pathlib import Path

from loguru import logger

from src.ingestion.chunking import ChunkedDocument, chunk_document
from src.ingestion.embeddings import embed_and_store
from src.ingestion.loaders import LoadedDocument, load_document


def process_document(file_path: str | Path) -> tuple[LoadedDocument, ChunkedDocument, int]:
    """Full ingestion pipeline: load -> chunk -> embed -> store.

    Returns (loaded_doc, chunked_doc, num_chunks_stored).
    """
    logger.info(f"Starting ingestion pipeline for: {file_path}")

    doc = load_document(file_path)
    chunked = chunk_document(doc)
    stored = embed_and_store(chunked)

    logger.info(f"Pipeline complete: {stored} chunks stored for '{doc.filename}'")
    return doc, chunked, stored
