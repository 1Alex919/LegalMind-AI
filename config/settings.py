from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # OpenAI
    OPENAI_API_KEY: str = ""
    OPENAI_CHAT_MODEL: str = "gpt-4o-mini"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"

    # ChromaDB
    CHROMA_PERSIST_DIR: str = "./data/chroma_db"
    CHROMA_COLLECTION_NAME: str = "legalmind_docs"

    # RAG Parameters
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    RETRIEVAL_TOP_K: int = 5
    HYBRID_ALPHA: float = 0.6  # weight for vector search (1 - alpha = BM25 weight)

    # Reranker
    RERANKER_MODEL: str = "rank-T5-flan"
    RERANKER_TOP_N: int = 3

    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # Logging
    LOG_LEVEL: str = "INFO"

    # Evaluation
    RAGAS_JUDGE_MODEL: str = "gpt-4o-mini"

    # Ollama (optional fallback)
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.1:8b"

    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    SAMPLE_CONTRACTS_DIR: Path = DATA_DIR / "sample_contracts"


settings = Settings()
