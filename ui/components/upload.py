"""File upload widget for Streamlit."""

import tempfile
from pathlib import Path

import streamlit as st

from src.ingestion import process_document


def render_upload() -> None:
    """Render the file upload component."""
    st.subheader("Upload Contract")

    uploaded_file = st.file_uploader(
        "Drag and drop a PDF or DOCX file",
        type=["pdf", "docx"],
        help="Supported formats: PDF, DOCX",
    )

    if uploaded_file is not None and st.button("Process Document", type="primary"):
        with st.spinner("Processing document..."):
            # Save to temp file
            suffix = Path(uploaded_file.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            try:
                loaded_doc, chunked_doc, chunks_stored = process_document(tmp_path)

                st.session_state["document"] = {
                    "filename": uploaded_file.name,
                    "total_pages": loaded_doc.total_pages,
                    "chunks_stored": chunks_stored,
                    "full_text": loaded_doc.full_text,
                    "pages": [
                        {"page": p.page_number, "text": p.text}
                        for p in loaded_doc.pages
                    ],
                }

                st.success(
                    f"Processed **{uploaded_file.name}**: "
                    f"{loaded_doc.total_pages} pages, "
                    f"{chunks_stored} chunks indexed"
                )

            except Exception as e:
                st.error(f"Processing failed: {e}")
            finally:
                Path(tmp_path).unlink(missing_ok=True)

    # Show document info if loaded
    doc = st.session_state.get("document")
    if doc:
        st.divider()
        col1, col2, col3 = st.columns(3)
        col1.metric("File", doc["filename"])
        col2.metric("Pages", doc["total_pages"])
        col3.metric("Chunks", doc["chunks_stored"])
