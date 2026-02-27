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
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from app.models.loader import get_reranker_model_path
from app.core.config import get_device

# [V3.1] 리랭커 기본 설정값 (NameError 방지)
DEFAULT_TOP_N = 5
DEFAULT_THRESHOLD = 0.5

# ── 전역 리랭커 인스턴스 (Singleton) ──
GLOBAL_RERANKER = None

def load_reranker_singleton(model_name: str = None):
    """
    리랭커 모델을 전역적으로 1회 로드하여 GPU 메모리에 고정합니다.
    서버 시작 시 호출되어 하이브리드 검색의 지연 시간을 최소화합니다.
    """
    global GLOBAL_RERANKER
    if GLOBAL_RERANKER is not None:
        return GLOBAL_RERANKER

    model_path = model_name or get_reranker_model_path()
    device = get_device()
    
    print(f"\n📦 [Reranker] Loading Singleton Model: {model_path}")
    print(f"🚀 [Reranker] Forcing Device: {device.upper()}")
    
    GLOBAL_RERANKER = HuggingFaceCrossEncoder(
        model_name=str(model_path), 
        model_kwargs={"local_files_only": True, "device": device}
    )
    print("✅ [Reranker] Singleton Model Loaded and Fixed on Hardware.\n")
    return GLOBAL_RERANKER

def build_reranker_retriever(
    base_retriever,
    model_name: str = None,
    top_n: int = DEFAULT_TOP_N,
) -> ContextualCompressionRetriever:
    """
    전역 싱글톤 리랭커를 사용하여 압축 리트리버를 생성합니다.
    """
    cross_encoder = load_reranker_singleton(model_name)
    compressor = CrossEncoderReranker(model=cross_encoder, top_n=top_n)
    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever,
    )

def rerank_documents(
    query: str,
    documents: List[Document],
    model_name: str = None,
    top_n: int = DEFAULT_TOP_N,
    threshold: float = DEFAULT_THRESHOLD,
) -> Tuple[List[Document], bool, float]:
    """
    전역 싱글톤 리랭커를 사용하여 문서들을 정밀 재정렬합니다.
    """
    if not documents:
        print("[Reranker] 입력 문서가 없습니다 → 관련 없음 반환")
        return [], False, 0.0

    # 싱글톤 모델 가져오기 (이미 로드되어 있어야 함)
    cross_encoder = load_reranker_singleton(model_name)

    print(f"[Reranker] {len(documents)}개 문서 재정렬 시작 (On {get_device().upper()})")
    print(f"[Reranker] 임계값: {threshold} | 최대 반환: {top_n}개")

    pairs = [(query, doc.page_content) for doc in documents]
    
    # [GPU 연산 강제] 모델이 device에 올라가 있음을 보장
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
        return [], False, float(scored[0][0]) if scored else 0.0

    top_docs = [doc for _, doc in passed[:top_n]]
    top_score = float(passed[0][0]) if passed else 0.0
    print(f"[Reranker] ✅ 임계값 통과: {len(passed)}개 → 상위 {len(top_docs)}개 반환 (최고 점수: {top_score:.4f})")
    return top_docs, True, top_score
