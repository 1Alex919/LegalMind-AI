"""OpenAI embeddings wrapper with ChromaDB storage."""

from pathlib import Path

import chromadb
from langchain_openai import OpenAIEmbeddings
from loguru import logger

from config.settings import settings
from src.ingestion.chunking import ChunkedDocument

# Instruction prefixes improve retrieval quality by telling the model
# what the text will be used for. While OpenAI embeddings don't require
# instructions like E5/BGE models, prepending context still helps
# differentiate query vs document embeddings in practice.
QUERY_PREFIX = "Represent this legal question for retrieving relevant contract clauses: "
DOCUMENT_PREFIX = ""  # documents are embedded as-is


def get_embedding_function() -> OpenAIEmbeddings:
    """Create OpenAI embeddings function for document storage."""
    return OpenAIEmbeddings(
        model=settings.OPENAI_EMBEDDING_MODEL,
        openai_api_key=settings.OPENAI_API_KEY,
    )


def get_query_embedding_function() -> OpenAIEmbeddings:
    """Create OpenAI embeddings function for queries with instruction prefix."""
    return _QueryPrefixEmbeddings(
        model=settings.OPENAI_EMBEDDING_MODEL,
        openai_api_key=settings.OPENAI_API_KEY,
    )


class _QueryPrefixEmbeddings(OpenAIEmbeddings):
    """OpenAI embeddings that prepend an instruction prefix to queries."""

    def embed_query(self, text: str) -> list[float]:
        return super().embed_query(QUERY_PREFIX + text)


def get_chroma_client() -> chromadb.ClientAPI:
    """Get persistent ChromaDB client."""
    persist_dir = Path(settings.CHROMA_PERSIST_DIR)
    persist_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(persist_dir))


def get_or_create_collection(
    client: chromadb.ClientAPI, name: str | None = None
) -> chromadb.Collection:
    """Get or create a ChromaDB collection."""
    collection_name = name or settings.CHROMA_COLLECTION_NAME
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def embed_and_store(chunked_doc: ChunkedDocument) -> int:
    """Embed child chunks and store in ChromaDB.

    Returns the number of chunks stored.
    """
    client = get_chroma_client()
    collection = get_or_create_collection(client)
    embeddings_fn = get_embedding_function()

    # Check for already-ingested documents
    existing = collection.get(where={"filename": chunked_doc.filename})
    if existing["ids"]:
        logger.warning(
            f"Document '{chunked_doc.filename}' already ingested "
            f"({len(existing['ids'])} chunks). Skipping."
        )
        return 0

    child_chunks = chunked_doc.child_chunks
    if not child_chunks:
        logger.warning("No child chunks to embed.")
        return 0

    texts = [c.text for c in child_chunks]
    ids = [c.chunk_id for c in child_chunks]
    metadatas = [c.metadata for c in child_chunks]

    # Embed in batches to avoid memory issues
    batch_size = 100
    total_stored = 0

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_ids = ids[i : i + batch_size]
        batch_meta = metadatas[i : i + batch_size]

        logger.info(
            f"Embedding batch {i // batch_size + 1} "
            f"({len(batch_texts)} chunks)..."
        )
        batch_embeddings = embeddings_fn.embed_documents(batch_texts)

        collection.add(
            ids=batch_ids,
            embeddings=batch_embeddings,
            documents=batch_texts,
            metadatas=batch_meta,
        )
        total_stored += len(batch_texts)

    # Store parent chunks in a separate collection for context expansion
    parent_collection = get_or_create_collection(client, name="legalmind_parents")
    parent_chunks = chunked_doc.parent_chunks

    if parent_chunks:
        parent_collection.add(
            ids=[c.chunk_id for c in parent_chunks],
            documents=[c.text for c in parent_chunks],
            metadatas=[c.metadata for c in parent_chunks],
        )

    logger.info(
        f"Stored {total_stored} child chunks and "
        f"{len(parent_chunks)} parent chunks for '{chunked_doc.filename}'"
    )
    return total_stored
