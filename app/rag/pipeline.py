# ============================================================
# [통합 Advanced RAG 파이프라인] — Advanced RAG Step 4
# ============================================================
# 흐름: 재작성된 쿼리 → Hybrid Retriever (Vector+BM25) → Reranker → 압축 Context
#
# 필수 Import:
#   langchain-community   -- BM25Retriever, EnsembleRetriever, Chroma
#   langchain-openai      -- OpenAIEmbeddings
#   sentence-transformers -- CrossEncoderReranker 백엔드
# ============================================================
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document

from app.rag.retriever import build_hybrid_retriever
from app.rag.reranker import rerank_documents

# ── Advanced RAG 통합 함수 ──
def get_advanced_retriever(
    documents: List[Document],
    vector_weight: float = 0.6,
    bm25_weight: float = 0.4,
    k: int = 10,
    rerank_top_n: int = 4,
    reranker_model: str = "BAAI/bge-reranker-v2-m3",
):
    """
    Hybrid Retriever + Cross-Encoder Reranker를 결합한
    통합 Advanced RAG 검색기를 반환합니다.

    Args:
        documents:       검색 대상 문서 목록
        vector_weight:   의미 검색 가중치 (기본 0.6)
        bm25_weight:     키워드 검색 가중치 (기본 0.4)
        k:               각 검색기당 반환 문서 수 (리랭킹 전)
        rerank_top_n:    리랭킹 후 최종 반환 문서 수
        reranker_model:  Cross-Encoder 모델명

    Returns:
        dict: {hybrid_retriever, reranker_config} — 검색 실행용 메타 정보
    """
    hybrid_retriever = build_hybrid_retriever(
        documents=documents,
        vector_weight=vector_weight,
        bm25_weight=bm25_weight,
        k=k,
    )
    return {
        "hybrid_retriever": hybrid_retriever,
        "reranker_model": reranker_model,
        "rerank_top_n": rerank_top_n,
    }

def run_advanced_rag(
    query: str,
    domain: str,
    documents: List[Document],
    filters: Optional[Dict[str, Any]] = None,
    vector_weight: float = 0.6,
    bm25_weight: float = 0.4,
    k: int = 10,
    rerank_top_n: int = 4,
) -> str:
    """
    완전한 Advanced RAG 파이프라인을 실행하여 압축된 Context 문자열을 반환합니다.
    전문가 에이전트(Specialist Nodes)에서 호출합니다.

    흐름:
        query (재작성된 쿼리)
            → Hybrid Retriever (Vector + BM25, 각 k개)
            → 도메인 필터링 (선택)
            → Cross-Encoder Reranker (top_n개로 압축)
            → Context 문자열 결합

    Returns:
        str: 압축된 관련 문서 텍스트 (LLM에게 전달할 Context)
    """
    print(f"[Advanced RAG] 쿼리: '{query}' | 도메인: {domain}")

    # 1. 하이브리드 검색 실행
    hybrid = build_hybrid_retriever(
        documents=documents,
        vector_weight=vector_weight,
        bm25_weight=bm25_weight,
        k=k,
    )
    retrieved_docs: List[Document] = hybrid.invoke(query)
    print(f"[Advanced RAG] 하이브리드 검색 결과: {len(retrieved_docs)}개")

    # 2. 도메인 필터 (메타데이터 기반 후처리)
    if domain:
        domain_lower = domain.lower()
        filtered = [
            doc for doc in retrieved_docs
            if doc.metadata.get("domain", "").lower() == domain_lower
        ]
        # 도메인 필터 후 최소 1개 이상 있으면 필터 적용, 없으면 전체 유지
        retrieved_docs = filtered if filtered else retrieved_docs
        print(f"[Advanced RAG] 도메인 필터({domain}) 후: {len(retrieved_docs)}개")

    # 3. Cross-Encoder 리랭킹 — [FIX] rerank_documents는 (docs, is_found) 튜플 반환
    top_docs, is_found = rerank_documents(
        query=query,
        documents=retrieved_docs,
        top_n=rerank_top_n,
    )

    # 4. 최종 Context 결합
    if not is_found or not top_docs:
        return "(관련 매뉴얼 문서를 찾을 수 없습니다.)"

    context = "\n\n---\n\n".join([
        f"[출처: {doc.metadata.get('source','?')} | {doc.metadata.get('chapter_path','?')}]\n{doc.page_content}"
        for doc in top_docs
    ])
    # 메타데이터를 Context 머리말에 삽입 → LLM이 [출처:...] 태그를 답변에 그대로 사용 가능

    print(f"[Advanced RAG] 최종 Context 길이: {len(context)}자 ({len(top_docs)}개 청크)")
    return context

# ── 레거시 단순 RAG 파이프라인 (기존 Specialist 호환용) ──
class PGVectorRetriever:
    """
    AWS RDS pgvector 기반 검색기의 Placeholder 클래스입니다.
    run_rag_pipeline() 레거시 함수에서 사용됩니다.
    Advanced RAG를 사용하려면 run_advanced_rag()를 직접 호출하세요.
    """
    def retrieve(self, query: str, metadata_filters: Dict[str, Any] = None) -> List[str]:
        print(f"[RAG] PGVector (Placeholder) 문서 검색: '{query}'")
        if metadata_filters:
            print(f"[RAG] 메타데이터 필터: {metadata_filters}")
        # TODO: 실제 AWS RDS pgvector 연결 및 유사도 검색 구현
        return [
            "[HD Hi5 Robot | 배터리 유지보수]\nHi5 제어기 Error E012: 서보 모터 엔코더 배터리 전압이 낮습니다.",
            "[HD Hi6 Robot | 케이블 점검]\n유지보수 가이드: 링 케이블과 커넥터의 파손 및 마모를 주기적으로 점검하십시오.",
        ]

def run_rag_pipeline(query: str, domain: str, filters: Dict[str, Any] = None) -> str:
    """
    레거시 단순 RAG 파이프라인 (Placeholder 기반).
    Advanced RAG 도입 전 호환성을 위해 유지합니다.
    실제 운영에서는 run_advanced_rag()로 전환하세요.
    """
    retriever = PGVectorRetriever()
    if domain:
        filters = filters or {}
        filters["domain"] = domain

    retrieved_docs = retriever.retrieve(query, metadata_filters=filters)
    context = "\n\n---\n\n".join(retrieved_docs)
    return context
