import os
import json
from tempfile import NamedTemporaryFile

from fastapi import APIRouter, Depends, Form, File, HTTPException, UploadFile, BackgroundTasks
from pydantic import BaseModel

from app.api.auth_api import get_me
from app.services.pdf_ingestion_service import incremental_embed_markdown_to_pgvector
from app.services.pdf_parse_service import parse_pdf_to_markdown

router = APIRouter(prefix="/admin", tags=["Admin Operations"])

class PreviewResponse(BaseModel):
    message: str
    parser_used: str | None = None
    markdown_text: str
    metadata_json: str
    already_exists: bool = False

class UploadResponse(BaseModel):
    message: str
    status: str = "success"
    parser_used: str | None = None
    local_md_path: str | None = None
    db_chunks_inserted: int
    total_chunks_parsed: int = 0
    db_chunks_skipped: int = 0
    db_chunks_deleted: int = 0
    file_hash: str | None = None


@router.post("/parse_pdf_preview", response_model=PreviewResponse)
async def parse_pdf_preview(
    file: UploadFile = File(...),
    parser: str = Form("marker"),
    admin_name: str = Form("admin"),
    current_user: dict = Depends(get_me),
):
    """
    Step 1: Parse PDF to markdown (marker/pypdf fallback) and return preview payload.
    Does NOT incrementaly embed or save into DB yet.
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
        from app.services.pdf_ingestion_service import get_all_registry_documents
        try:
            docs = get_all_registry_documents()
            already_exists = any(d["source_key"] == filename for d in docs)
        except Exception:
            already_exists = False

        full_text, parser_used, metadata = parse_pdf_to_markdown(
            tmp_pdf_path, 
            source_name=filename, 
            parser_choice=parser,
            creator=admin_name
        )
        return PreviewResponse(
            message=f"Successfully parsed {filename} for preview.",
            parser_used=parser_used,
            markdown_text=full_text,
            metadata_json=json.dumps(metadata, ensure_ascii=False, indent=2),
            already_exists=already_exists
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF Parsing Preview Error: {e}")
    finally:
        if os.path.exists(tmp_pdf_path):
            os.remove(tmp_pdf_path)


@router.post("/upload_pdf", response_model=UploadResponse)
async def upload_pdf(
    file: UploadFile = File(...),
    parser: str = Form("marker"),
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
        admin_name = (current_user.get("username") if isinstance(current_user, dict) else None) or "admin"
        full_text, parser_used, metadata = parse_pdf_to_markdown(
            tmp_pdf_path, 
            source_name=filename,
            parser_choice=parser,
            creator=admin_name
        )

        import json
        ingest_result = incremental_embed_markdown_to_pgvector(
            markdown_text=full_text,
            file_bytes=content,
            original_filename=filename,
            creator=admin_name,
            s3_pdf_key=None,
            s3_md_key=None,
            file_metadata_json=json.dumps(metadata, ensure_ascii=False)
        )

        # [V4.1] 강제 초기화(Refresh) 연동: 
        # DB에 변화가 생겼다면(삽입/삭제), 캐싱된 BM25 싱글톤을 삭제하고 재생성합니다.
        chunks_inserted = ingest_result.get("db_chunks_inserted", 0)
        chunks_deleted = ingest_result.get("db_chunks_deleted", 0)
        
        if chunks_inserted > 0 or chunks_deleted > 0:
            try:
                # Use quasi-incremental update directly instead of full refresh if we wanted to
                from app.rag.retriever import update_bm25_cache_for_uploaded_source
                update_bm25_cache_for_uploaded_source([filename])
            except Exception as e:
                print(f"⚠️ BM25 Refresh Error: {e}")

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


from infrastructure.aws.s3_utils import S3Client
from datetime import datetime
from app.services.pdf_ingestion_service import get_all_registry_documents, delete_registry_documents_by_sources
from typing import List

class DeleteRequest(BaseModel):
    source_keys: List[str]


@router.post("/commit_pdf", response_model=UploadResponse)
async def commit_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    markdown_text: str = Form(...),
    metadata_json: str = Form(...),
    admin_name: str = Form(...),
    current_user: dict = Depends(get_me),
):
    """
    Step 2: Commit PDF, parsed MD, and JSON to S3, then incrementally embed into PGVector.
    """
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin privileges required")

    filename = file.filename or "unknown.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded PDF is empty")

    date_str = datetime.now().strftime("%Y%m%d")
    base_prefix = f"ingested_docs/{date_str}_{admin_name}_{filename}"
    s3_pdf_key = f"{base_prefix}/{filename}"
    s3_md_key = f"{base_prefix}/{filename}.md"
    s3_json_key = f"{base_prefix}/{filename}.json"

    # Upload to S3
    try:
        s3_client = S3Client()
        s3_client.upload_file_bytes(content, s3_pdf_key, "application/pdf")
        s3_client.upload_file_bytes(markdown_text.encode("utf-8"), s3_md_key, "text/markdown")
        s3_client.upload_file_bytes(metadata_json.encode("utf-8"), s3_json_key, "application/json")
    except Exception as e:
        print(f"S3 Upload failed: {e}")
        # Proceed with embedding even if S3 fails, or fail hard? For now, print error but try to continue,
        # actually, S3 is required according to instructions.
        raise HTTPException(status_code=500, detail=f"S3 Upload Error: {e}")

    try:
        ingest_result = incremental_embed_markdown_to_pgvector(
            markdown_text=markdown_text,
            file_bytes=content,
            original_filename=filename,
            creator=admin_name,
            s3_pdf_key=s3_pdf_key,
            s3_md_key=s3_md_key,
            file_metadata_json=metadata_json,
        )

        chunks_inserted = ingest_result.get("db_chunks_inserted", 0)
        chunks_deleted = ingest_result.get("db_chunks_deleted", 0)
        
        if chunks_inserted > 0 or chunks_deleted > 0:
            from app.rag.retriever import update_bm25_cache_for_uploaded_source
            background_tasks.add_task(update_bm25_cache_for_uploaded_source, [filename])

        return UploadResponse(
            message=f"Successfully committed {filename} to S3 and PGVector.",
            status="background_processing",
            parser_used="pre-parsed",
            local_md_path=ingest_result["local_md_path"],
            db_chunks_inserted=ingest_result["db_chunks_inserted"],
            total_chunks_parsed=ingest_result["total_chunks_parsed"],
            db_chunks_skipped=ingest_result["db_chunks_skipped"],
            db_chunks_deleted=ingest_result["db_chunks_deleted"],
            file_hash=ingest_result["file_hash"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF Commit Pipeline Error: {e}")


@router.get("/registry")
async def get_registry(current_user: dict = Depends(get_me)):
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin privileges required")
        
    try:
        docs = get_all_registry_documents()
        return {"documents": docs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch registry: {e}")


@router.delete("/registry")
async def delete_registry(req: DeleteRequest, background_tasks: BackgroundTasks, current_user: dict = Depends(get_me)):
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin privileges required")
        
    try:
        # 1. Delete from PGVector and Registry DB
        result = delete_registry_documents_by_sources(req.source_keys)
        
        # 2. Delete from S3
        s3_keys = result.get("s3_keys_to_delete", [])
        if s3_keys:
            s3_client = S3Client()
            del_res = s3_client.delete_files(s3_keys)
            if del_res.get("Errors"):
                print(f"⚠️ Failed to delete some S3 files: {del_res['Errors']}")
            
        # 3. Refresh BM25 Cache in Background
        if result.get("deleted_chunks", 0) > 0:
            try:
                from app.rag.retriever import update_bm25_cache_for_uploaded_source
                background_tasks.add_task(update_bm25_cache_for_uploaded_source, req.source_keys)
            except Exception as e:
                print(f"⚠️ Background task scheduling error for BM25: {e}")
                
        return {
            "message": f"Successfully deleted {result['deleted_chunks']} chunks and associated files.",
            "status": "background_processing",
            "deleted_chunks": result['deleted_chunks'],
            "deleted_s3_keys": s3_keys
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete registry documents: {e}")
