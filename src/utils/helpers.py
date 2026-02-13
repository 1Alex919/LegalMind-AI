from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    """Create directory if it doesn't exist and return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_file_extension(filename: str) -> str:
    """Return lowercase file extension without the dot."""
    return Path(filename).suffix.lstrip(".").lower()


def is_supported_file(filename: str) -> bool:
    """Check if the file type is supported for ingestion."""
    supported = {"pdf", "docx"}
    return get_file_extension(filename) in supported
