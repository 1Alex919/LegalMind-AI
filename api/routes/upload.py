"""Document upload endpoint."""

import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile
from loguru import logger

from api.schemas import UploadResponse
from config.settings import settings
from src.ingestion import process_document

router = APIRouter()

# In-memory document registry (maps document_id -> filename)
_document_registry: dict[str, str] = {}


def get_document_registry() -> dict[str, str]:
    return _document_registry


@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile) -> UploadResponse:
    """Upload a PDF or DOCX document for analysis."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    ext = Path(file.filename).suffix.lower()
    if ext not in (".pdf", ".docx"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Use .pdf or .docx",
        )

    # Save uploaded file
    document_id = str(uuid.uuid4())
    upload_dir = Path(settings.DATA_DIR) / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / f"{document_id}{ext}"

    try:
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        logger.info(f"Uploaded {file.filename} as {document_id}")

        # Process document through ingestion pipeline
        loaded_doc, chunked_doc, chunks_stored = process_document(file_path)

        _document_registry[document_id] = file.filename

        return UploadResponse(
            document_id=document_id,
            filename=file.filename,
            total_pages=loaded_doc.total_pages,
            chunks_stored=chunks_stored,
            message=f"Document processed: {chunks_stored} chunks indexed",
        )

    except Exception as e:
        logger.error(f"Upload failed: {e}")
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")
