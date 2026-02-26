import os
from tempfile import NamedTemporaryFile

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.api.auth_api import get_me
from app.services.pdf_ingestion_service import incremental_embed_markdown_to_pgvector
from app.services.pdf_parse_service import parse_pdf_to_markdown

router = APIRouter(prefix="/admin", tags=["Admin Operations"])


class UploadResponse(BaseModel):
    message: str
    parser_used: str | None = None
    local_md_path: str | None = None
    db_chunks_inserted: int
    total_chunks_parsed: int = 0
    db_chunks_skipped: int = 0
    db_chunks_deleted: int = 0
    file_hash: str | None = None


@router.post("/upload_pdf", response_model=UploadResponse)
async def upload_pdf(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_me),
):
    """
    Admin PDF pipeline (local):
    1) parse PDF to markdown (marker/pypdf fallback)
    2) incremental chunk embedding into PGVector
    """
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin privileges required")

    filename = file.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded PDF is empty")

    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        tmp_pdf.write(content)
        tmp_pdf_path = tmp_pdf.name

    try:
        full_text, parser_used = parse_pdf_to_markdown(tmp_pdf_path, source_name=filename)

        ingest_result = incremental_embed_markdown_to_pgvector(
            markdown_text=full_text,
            file_bytes=content,
            original_filename=filename,
            creator=(current_user.get("username") if isinstance(current_user, dict) else None) or "admin",
            s3_pdf_key=None,
            s3_md_key=None,
        )

        return UploadResponse(
            message=f"Successfully parsed {filename}. {ingest_result['message_suffix']}",
            parser_used=parser_used,
            local_md_path=ingest_result["local_md_path"],
            db_chunks_inserted=ingest_result["db_chunks_inserted"],
            total_chunks_parsed=ingest_result["total_chunks_parsed"],
            db_chunks_skipped=ingest_result["db_chunks_skipped"],
            db_chunks_deleted=ingest_result["db_chunks_deleted"],
            file_hash=ingest_result["file_hash"],
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF Processing Pipeline Error: {e}")
    finally:
        if os.path.exists(tmp_pdf_path):
            os.remove(tmp_pdf_path)
