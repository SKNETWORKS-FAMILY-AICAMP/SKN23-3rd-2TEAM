# ============================================================
# [하이브리드 검색기] Hybrid Retriever — Advanced RAG Step 2
# ============================================================
# 역할: 의미(Semantic) 검색과 키워드(Keyword) 검색을 융합하여
#       에러 코드/부품명 누락 없는 정확한 문서를 검색합니다.
#
# 필수 Import (pyproject.toml에 추가 필요):
#   langchain-community   -- BM25Retriever, EnsembleRetriever
#   rank_bm25             -- BM25 백엔드 라이브러리 (pip install rank-bm25)
#   langchain-openai      -- OpenAIEmbeddings
# ============================================================
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document

# ── VectorStore 기반 의미 검색 (pgvector 연동) ──
# 실제 운영 시 langchain_community.vectorstores.PGVector로 교체
from langchain_community.vectorstores import Chroma          # 로컬 테스트용
from langchain_openai import OpenAIEmbeddings

# ── BM25 키워드 검색 (에러 코드, 부품명 정확 매칭용) ──
from langchain_community.retrievers import BM25Retriever

# ── 두 검색기 결합 ──
from langchain.retrievers import EnsembleRetriever

def build_vector_retriever(
    documents: Optional[List[Document]] = None,
    k: int = 10,
):
    """
    pgvector (의미 기반) Retriever를 생성합니다.
    실제 운영 시 PGVector 커넥션 스트링으로 교체하세요.

    Args:
        documents: 초기 벡터 저장소 구성용 문서 (테스트 시 활용)
        k: 검색할 최대 문서 수
    """
    embeddings = OpenAIEmbeddings()
    # text-embedding-ada-002 모델로 쿼리와 문서를 벡터화

    if documents:
        # 로컬 Chroma DB로 빠른 테스트 (추후 PGVector로 교체)
        vectorstore = Chroma.from_documents(documents, embeddings)
    else:
        # 운영 환경: AWS RDS pgvector 연결
        # 예시 (실제 구현 시 주석 해제):
        # from langchain_community.vectorstores import PGVector
        # conn_str = "postgresql+psycopg2://user:password@host:5432/dbname"
        # vectorstore = PGVector(connection_string=conn_str, embedding_function=embeddings,
        #                        collection_name="industrial_manuals")
        raise NotImplementedError("운영 환경의 PGVector 연결 설정이 필요합니다.")

    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
        # search_type="mmr" → 다양성 극대화 검색 (옵션)
    )

def build_bm25_retriever(
    documents: List[Document],
    k: int = 10,
) -> BM25Retriever:
    """
    BM25 Keyword Retriever를 생성합니다.
    에러 코드(E012, 4107), 부품명(TP630, YRC1000) 등
    정확한 배치 매칭이 필요한 경우에 강점을 발휘합니다.

    Args:
        documents: BM25 색인을 구성할 문서 목록
        k: 검색할 최대 문서 수
    """
    retriever = BM25Retriever.from_documents(documents)
    retriever.k = k
    return retriever

def build_hybrid_retriever(
    documents: List[Document],
    vector_weight: float = 0.6,
    bm25_weight: float = 0.4,
    k: int = 10,
) -> EnsembleRetriever:
    """
    Vector(Semantic) + BM25(Keyword) 검색을 EnsembleRetriever로 결합합니다.

    Args:
        documents:      BM25 색인 및 Vector DB 구성용 문서
        vector_weight:  의미 검색 가중치 (기본 0.6)
        bm25_weight:    키워드 검색 가중치 (기본 0.4)
        k:              각 검색기당 반환할 문서 수

    Returns:
        EnsembleRetriever: 두 검색기를 결합한 하이브리드 검색기
    """
    print(f"[Hybrid Retriever] 빌드 시작 (vector:{vector_weight} / bm25:{bm25_weight})")

    vector_retriever = build_vector_retriever(documents=documents, k=k)
    bm25_retriever = build_bm25_retriever(documents=documents, k=k)

    ensemble = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[vector_weight, bm25_weight]
        # 가중치 합은 반드시 1.0이어야 합니다
        # 에러 코드 매칭이 중요한 경우 bm25_weight를 높이세요
    )

    print(f"[Hybrid Retriever] 빌드 완료")
    return ensemble
