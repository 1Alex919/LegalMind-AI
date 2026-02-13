# Architecture

## System Overview

LegalMind AI is a RAG-based legal document analysis system built with a modular, multi-agent architecture.

## Components

### 1. Document Ingestion Pipeline

```
PDF/DOCX -> Loaders -> Parent-Child Chunker -> OpenAI Embeddings -> ChromaDB
```

- **Loaders** (`src/ingestion/loaders.py`): Extract text from PDF (pypdf) and DOCX (python-docx). PDFs are split by page; DOCX by ~3000-char sections.
- **Chunking** (`src/ingestion/chunking.py`): Parent-child strategy. Parent chunks (~1536 chars) provide broad context. Child chunks (~512 chars) are used for precise retrieval. Each child references its parent for context expansion.
- **Embeddings** (`src/ingestion/embeddings.py`): OpenAI `text-embedding-3-small`. Batch processing (100 at a time). Deduplication by filename to avoid re-embedding.

### 2. Hybrid RAG Pipeline

```
Query -> [Query Expansion] -> BM25 + Vector Search -> Score Fusion -> Reranking -> Parent Context
```

- **Hybrid Search** (`src/retrieval/hybrid_search.py`): Combines BM25 sparse retrieval with ChromaDB dense vector search. Scores are min-max normalized then combined with configurable alpha weight (default 0.6 vector / 0.4 BM25).
- **Reranker** (`src/retrieval/reranker.py`): FlashRank (rank-T5-flan) reranks top candidates. Falls back to hybrid scores on failure.
- **Query Expansion** (`src/retrieval/query_expansion.py`): HyDE generates hypothetical contract passages. Multi-query generates 3 alternative phrasings.
- **Parent Document Retrieval**: After finding relevant child chunks, fetches parent chunks from a separate collection for richer context.

### 3. Multi-Agent System

```
Input -> Task Classifier -> Orchestrator (LangGraph) -> Agent -> Structured Output
```

- **Orchestrator** (`src/agents/orchestrator.py`): LangGraph `StateGraph` with nodes for classification, retrieval, and three agent paths. Auto-classifies tasks if type not specified.
- **Risk Agent**: Returns structured JSON with risk type, severity, clause text, explanation, and recommendation.
- **Q&A Agent**: Answers questions with source citations and confidence scores. Handles "I don't know" gracefully.
- **Summary Agent**: Generates 5-7 sentence summaries with key points, parties, and contract type.

### 4. Explainability Layer

Every response includes:
- **Source tracking**: Which chunks were used, with page numbers and relevance scores
- **Confidence scoring**: Combines retrieval score mean, consistency, and LLM self-assessment
- **Reasoning trace**: Step-by-step log of the pipeline with timing

### 5. LLM Provider

- **Primary**: OpenAI GPT-4o-mini with tenacity retry (exponential backoff, 3 attempts)
- **Fallback**: Ollama (local) via OpenAI-compatible API. Auto-detected at startup.
- **Rate Limiting**: Sliding window (50 req/min). Blocks and waits when limit reached.

## Data Flow

1. User uploads document via UI or API
2. Ingestion pipeline extracts, chunks, embeds, and stores
3. User submits a task (risk/qa/summary)
4. Orchestrator classifies task and retrieves relevant chunks
5. Appropriate agent processes context with LLM
6. Explainability layer enriches response with sources and confidence
7. Structured JSON returned to client

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Embedding model | text-embedding-3-small | Cost-effective, good quality |
| Vector DB | ChromaDB (embedded) | No server needed, persistent |
| Reranker | FlashRank | ~100MB vs ~560MB for BGE |
| Chunking | Parent-child | Better context than flat chunking |
| Agent framework | LangGraph | Explicit state graph, debuggable |
| Score fusion | Weighted sum | Simple, effective, tunable |
