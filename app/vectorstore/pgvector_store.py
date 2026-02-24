# ============================================================
# [AWS RDS pgvector 벡터 스토어 구현] app/vectorstore/pgvector_store.py
# ============================================================
# 프로덕션 환경에서 벡터 검색을 위한 AWS RDS pgvector 연결을 구현합니다.
# 로컬 개발 시에는 chroma_store.py를 대신 사용하세요.
# ============================================================
import os
from typing import List, Optional
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

# TODO: RDS 연결 환경변수 설정 후 주석 해제
# from langchain_community.vectorstores import PGVector


class PGVectorStore:
    """
    AWS RDS PostgreSQL + pgvector 벡터 스토어.
    
    환경변수:
        RDS_HOST: RDS 엔드포인트 (예: chatbot-db.xxxxxxxx.ap-northeast-2.rds.amazonaws.com)
        RDS_DB: 데이터베이스 이름 (예: chatbot_db)
        RDS_USER: DB 사용자 이름 (예: postgres)
        RDS_PASSWORD: DB 비밀번호
        RDS_PORT: DB 포트 (기본: 5432)
    """
    
    def __init__(self, collection_name: str = "industrial_manuals"):
        self.collection_name = collection_name
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # RDS 연결 문자열 구성
        host = os.getenv("RDS_HOST", "localhost")
        db   = os.getenv("RDS_DB", "chatbot_db")
        user = os.getenv("RDS_USER", "postgres")
        pwd  = os.getenv("RDS_PASSWORD", "password")
        port = os.getenv("RDS_PORT", "5432")
        
        self.connection_string = (
            f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{db}"
        )
        self._store = None
    
    def _get_store(self):
        """지연 초기화: 필요 시점에 PGVector 연결 생성."""
        # TODO: langchain_community.vectorstores.PGVector 구현 완료 후 활성화
        # if self._store is None:
        #     from langchain_community.vectorstores import PGVector
        #     self._store = PGVector(
        #         connection_string=self.connection_string,
        #         embedding_function=self.embeddings,
        #         collection_name=self.collection_name,
        #     )
        # return self._store
        raise NotImplementedError(
            "PGVectorStore: RDS 연결 구현이 필요합니다. "
            "RDS_HOST, RDS_DB, RDS_USER, RDS_PASSWORD 환경변수를 설정 후 구현하세요."
        )
    
    def as_retriever(self, k: int = 10):
        """Retriever 인터페이스 반환 (RAG 파이프라인 연동용)."""
        store = self._get_store()
        return store.as_retriever(search_kwargs={"k": k})
    
    def add_documents(self, documents: List[Document]):
        """문서를 pgvector에 임베딩하여 저장합니다."""
        store = self._get_store()
        store.add_documents(documents)
        print(f"[PGVectorStore] {len(documents)}개 문서 저장 완료")
