import os
import io
import time
from pathlib import Path
from tempfile import NamedTemporaryFile
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from pydantic import BaseModel

# AWS S3 및 PgVector 의존성 임포트
from app.infrastructure.aws.s3_client import S3Client
from app.core.database import open_optional_ssh_tunnel, get_connection_kwargs
from app.api.auth_api import get_me  # JWT Token Validations
import psycopg2
from psycopg2.extras import execute_values

router = APIRouter(prefix="/admin", tags=["Admin Operations"])

S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "weld-bot-knowledge-base")
AWS_REGION = os.getenv("AWS_REGION", "ap-northeast-2")
s3 = S3Client(region=AWS_REGION)

# [Placeholder] 실제 Marker 라이브러리 연동
# poetry add marker-pdf
try:
    from marker.convert import convert_single_pdf
    from marker.models import load_all_models
    HAS_MARKER = True
    marker_models = load_all_models()
except ImportError:
    HAS_MARKER = False
    marker_models = None

class UploadResponse(BaseModel):
    message: str
    s3_pdf_path: str
    s3_md_path: str
    db_chunks_inserted: int

@router.post("/upload_pdf", response_model=UploadResponse)
async def upload_pdf(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_me)
):
    """
    1. 관리자 권한 확인
    2. PDF를 S3에 원본 저장
    3. Marker 모델로 PDF를 Markdown 파싱
    4. 결과를 S3에 저장
    5. Markdown을 Chunking 후 PgVector 등재
    """
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin privileges required")

    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    if not HAS_MARKER:
        # For environment where Marker is too heavy, bypass or raise error
        # raising error to indicate we need the marker-pdf package
        raise HTTPException(status_code=501, detail="marker-pdf is not installed in the backend.")

    content = await file.read()
    
    # 임시 파일로 PDF 저장 (Marker가 파일 패스를 주로 요구함)
    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        tmp_pdf.write(content)
        tmp_pdf_path = tmp_pdf.name

    try:
        # 1. 원본 PDF를 S3에 업로드
        s3_pdf_key = f"raw_pdfs/{int(time.time())}_{file.filename}"
        s3.client.upload_file(tmp_pdf_path, S3_BUCKET_NAME, s3_pdf_key)

        # 2. Marker 로 PDF -> Markdown 변환
        # convert_single_pdf returns (full_text, out_meta)
        full_text, out_meta = convert_single_pdf(tmp_pdf_path, marker_models)
        
        # 3. 파싱 결과를 S3에 저장
        s3_md_key = f"parsed_mds/{int(time.time())}_{file.filename.replace('.pdf', '.md')}"
        s3.client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=s3_md_key,
            Body=full_text.encode("utf-8")
        )

        # 4. Markdown 청킹 및 Text Embedding Vector Store 적재 로직
        # 여기서는 langchain_openai 의 OpenAIEmbeddings를 사용하는 파이프라인 호출 가정
        from app.ingest.chunking import get_text_splitter
        from langchain_openai import OpenAIEmbeddings
        from langchain.schema import Document
        from app.vectorstore.pgvector_store import get_vector_store, PGVectorStoreManager

        splitter = get_text_splitter()
        # Document 객체로 변환
        docs = [Document(page_content=full_text, metadata={"source": file.filename, "s3_pdf": s3_pdf_key, "s3_md": s3_md_key})]
        chunks = splitter.split_documents(docs)

        # VectorStore 연동하여 DB 저장 (run_tunnel 처리 포함)
        with PGVectorStoreManager() as _:
            vector_store = get_vector_store()
            # 실제로 DB에 적재
            vector_store.add_documents(chunks)

        return UploadResponse(
            message=f"Successfully parsed and ingested {file.filename}",
            s3_pdf_path=f"s3://{S3_BUCKET_NAME}/{s3_pdf_key}",
            s3_md_path=f"s3://{S3_BUCKET_NAME}/{s3_md_key}",
            db_chunks_inserted=len(chunks)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF Processing Pipeline Error: {str(e)}")
    finally:
        # 임시 파일 삭제
        if os.path.exists(tmp_pdf_path):
            os.remove(tmp_pdf_path)
