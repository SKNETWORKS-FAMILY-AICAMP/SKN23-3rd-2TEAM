# ============================================================
# [리랭커 세부 튜닝] app/rag/reranker.py
# ============================================================
# 고도화 내용:
#   1. 임계값(Threshold) 필터: score < threshold 문서 제거
#   2. 점수 없을 경우 "관련 없음" 상태 반환 → Verifier 자동 작동
#   3. BAAI/bge-reranker-v2-m3 (한국어+영어 동시 지원)
# ============================================================
from typing import List, Tuple
from langchain_core.documents import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# ── 임계 점수 상수 ──
DEFAULT_THRESHOLD = 0.5   # 이 점수 미만의 문서는 질문과 무관하다고 판단하여 제거
DEFAULT_TOP_N    = 4      # 임계값 통과 후 최종 반환 최대 문서 수

def build_reranker_retriever(
    base_retriever,
    model_name: str = "BAAI/bge-reranker-v2-m3",
    top_n: int = DEFAULT_TOP_N,
) -> ContextualCompressionRetriever:
    """
    base_retriever 결과를 Cross-Encoder로 재정렬하는 ContextualCompressionRetriever를 생성합니다.
    (threshold 필터 없이 LangChain 내장 압축만 적용합니다. 세밀한 제어는 rerank_documents 사용)
    """
    print(f"[Reranker] 모델 로드 중: {model_name} | top_{top_n}")
    cross_encoder = HuggingFaceCrossEncoder(model_name=model_name)
    compressor = CrossEncoderReranker(model=cross_encoder, top_n=top_n)
    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever,
    )

def rerank_documents(
    query: str,
    documents: List[Document],
    model_name: str = "BAAI/bge-reranker-v2-m3",
    top_n: int = DEFAULT_TOP_N,
    threshold: float = DEFAULT_THRESHOLD,
) -> Tuple[List[Document], bool]:
    """
    하이브리드 검색으로 올라온 문서들을 Cross-Encoder로 정밀 재정렬합니다.
    임계값(threshold) 미만의 문서를 과감히 제거하여 노이즈를 차단합니다.

    Args:
        query:      사용자의 (재작성된) 검색 쿼리
        documents:  하이브리드 검색 결과 문서 목록
        model_name: Cross-Encoder 모델 ID
        top_n:      임계값 통과 후 최종 반환할 최대 문서 수
        threshold:  최소 관련성 점수 (0.0 ~ 1.0). 미만 문서는 전부 폐기.

    Returns:
        (List[Document], bool):
            - List[Document]: 임계값 통과 + top_n 이내의 핵심 문서
            - bool: True이면 관련 문서 있음, False이면 "찾을 수 없음" 상태
    """
    if not documents:
        print("[Reranker] 입력 문서가 없습니다 → 관련 없음 반환")
        return [], False

    print(f"[Reranker] {len(documents)}개 문서 재정렬 시작")
    print(f"[Reranker] 임계값: {threshold} | 최대 반환: {top_n}개")

    # Cross-Encoder 모델 로드 및 점수 계산
    cross_encoder = HuggingFaceCrossEncoder(model_name=model_name)
    pairs = [(query, doc.page_content) for doc in documents]
    raw_scores = cross_encoder.score(pairs)
    # raw_scores는 로짓(logit) 값일 수 있어 sigmoid 변환 필요
    import math
    scores = [1 / (1 + math.exp(-s)) for s in raw_scores]

    # 점수와 문서를 묶어 정렬
    scored = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)

    # 점수 현황 로깅
    for i, (score, doc) in enumerate(scored):
        chapter = doc.metadata.get("chapter_path", "?")
        print(f"  [{i+1}] Score: {score:.4f} | {chapter}")

    # 임계값 필터 적용
    passed = [(score, doc) for score, doc in scored if score >= threshold]

    if not passed:
        print(f"[Reranker] ⚠️ 임계값({threshold}) 통과 문서 없음 → '관련 없음' 반환")
        print(f"           최고 점수: {scored[0][0]:.4f} (기준 미달)")
        # is_found=False → run_rag_pipeline에서 "(관련 매뉴얼 없음)" 반환
        # → verifier 작동 시 context 없음으로 Hallucination 처리 → fallback 유도
        return [], False

    top_docs = [doc for _, doc in passed[:top_n]]
    print(f"[Reranker] ✅ 임계값 통과: {len(passed)}개 → 상위 {len(top_docs)}개 반환")
    return top_docs, True
