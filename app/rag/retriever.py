import os
import pickle
import time
import re
import json
from typing import List

from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_core.documents import Document

from app.vectorstore.pgvector_store import PGVectorStoreManager

from app.core.config import (
    CACHE_DIR, COLLECTION_NAME, 
    DEFAULT_VECTOR_WEIGHT, DEFAULT_BM25_WEIGHT, TECHNICAL_BM25_WEIGHT
)

BM25_CACHE_PATH = CACHE_DIR / "bm25_retriever.pkl"
CACHED_BM25_RETRIEVER = None

def korean_custom_preprocess(text: str) -> List[str]:
    """
    [추가됨] 한글 조사 및 특수기호로 인해 영어/숫자 에러코드가 매칭되지 않는 현상 방지
    예: "M0042가" -> "M0042 가" 로 분리될 수 있도록 특수기호 정제 후 띄어쓰기 기준 분리
    """
    if not isinstance(text, str):
        return []
    # 한글, 영문, 숫자를 제외한 모든 특수기호를 공백으로 치환
    text = re.sub(r'[^가-힣A-Za-z0-9]', ' ', text)
    return text.split()


def _doc_fingerprint(doc: Document) -> str:
    payload = {
        "page_content": doc.page_content,
        "metadata": doc.metadata or {},
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _atomic_dump_bm25_cache(retriever: BM25Retriever) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = BM25_CACHE_PATH.with_suffix(BM25_CACHE_PATH.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(retriever, f)
    os.replace(tmp_path, BM25_CACHE_PATH)


def _load_cached_bm25_from_memory_or_file() -> BM25Retriever | None:
    global CACHED_BM25_RETRIEVER
    if CACHED_BM25_RETRIEVER is not None:
        return CACHED_BM25_RETRIEVER
    if BM25_CACHE_PATH.exists():
        with open(BM25_CACHE_PATH, "rb") as f:
            CACHED_BM25_RETRIEVER = pickle.load(f)
        return CACHED_BM25_RETRIEVER
    return None

def _load_or_create_bm25_retriever(vector_store):
    """
    BM25 리트리버를 메모리(Singleton) -> 캐시 파일 -> RDS 순으로 로드합니다.
    """
    global CACHED_BM25_RETRIEVER
    
    if CACHED_BM25_RETRIEVER is not None:
        return CACHED_BM25_RETRIEVER

    if not CACHE_DIR.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if BM25_CACHE_PATH.exists():
        print(f"📦 BM25 인덱스 캐시 로드 중 (File): {BM25_CACHE_PATH}")
        with open(BM25_CACHE_PATH, "rb") as f:
            CACHED_BM25_RETRIEVER = pickle.load(f)
            return CACHED_BM25_RETRIEVER
    
    print("⚠️ BM25 캐시가 없습니다. RDS에서 모든 문서를 로드하여 인덱스를 생성합니다.")
    start_time = time.time()
    
    try:
        # RDS에서 모든 문서 가져오기 (전량 로드)
        # 주의: 빈 문자열 검색이 일부 임베딩 모델에서 불안정할 경우 vector_store.get() 등 활용 권장
        all_docs = vector_store.similarity_search("", k=100000)
        print(f"📝 {len(all_docs)}개의 문서를 로드했습니다. BM25 인덱싱 시작...")
        
        # [수정됨] 커스텀 전처리 함수(korean_custom_preprocess) 적용
        bm25_retriever = BM25Retriever.from_documents(
            all_docs,
            preprocess_func=korean_custom_preprocess
        )
        bm25_retriever.k = 5
        
        # 캐시에 저장
        with open(BM25_CACHE_PATH, "wb") as f:
            pickle.dump(bm25_retriever, f)
            
        CACHED_BM25_RETRIEVER = bm25_retriever
        end_time = time.time()
        print(f"✅ BM25 인덱스 생성 및 캐시 저장 완료 ({end_time - start_time:.2f}초 소요)")
        return CACHED_BM25_RETRIEVER
        
    except Exception as e:
        print(f"❌ BM25 인덱스 생성 실패: {e}")
        raise e

def update_bm25_cache_for_uploaded_source(
    uploaded_docs: List[Document],
    *,
    source_file: str | None = None,
) -> dict[str, object]:
    """
    Incrementally update the local BM25 cache by replacing documents for one source_file.

    This avoids a full RDS download during admin uploads. It only reads/writes the local
    `bm25_retriever.pkl` and swaps the chunks for the uploaded file.
    """
    global CACHED_BM25_RETRIEVER

    clean_docs = [
        doc for doc in (uploaded_docs or [])
        if isinstance(doc, Document) and isinstance(doc.page_content, str) and doc.page_content.strip()
    ]
    if not clean_docs:
        return {
            "bm25_cache_updated": False,
            "bm25_status": "skipped_empty_upload",
            "bm25_total_docs": None,
            "bm25_source_docs": 0,
            "bm25_prev_source_docs": None,
        }

    if source_file is None:
        source_values = {
            str((doc.metadata or {}).get("source_file", "")).strip()
            for doc in clean_docs
        }
        source_values.discard("")
        if len(source_values) == 1:
            source_file = next(iter(source_values))

    source_file = (source_file or "").strip()
    if not source_file:
        return {
            "bm25_cache_updated": False,
            "bm25_status": "skipped_missing_source_file",
            "bm25_total_docs": None,
            "bm25_source_docs": len(clean_docs),
            "bm25_prev_source_docs": None,
        }

    cached = _load_cached_bm25_from_memory_or_file()
    if cached is None:
        return {
            "bm25_cache_updated": False,
            "bm25_status": "skipped_cache_not_found",
            "bm25_total_docs": None,
            "bm25_source_docs": len(clean_docs),
            "bm25_prev_source_docs": None,
        }

    cached_docs = list(getattr(cached, "docs", []) or [])
    unaffected_docs: list[Document] = []
    prev_source_docs: list[Document] = []
    for doc in cached_docs:
        meta = getattr(doc, "metadata", {}) or {}
        if str(meta.get("source_file", "")).strip() == source_file:
            prev_source_docs.append(doc)
        else:
            unaffected_docs.append(doc)

    prev_fingerprints = sorted(_doc_fingerprint(doc) for doc in prev_source_docs)
    new_fingerprints = sorted(_doc_fingerprint(doc) for doc in clean_docs)
    if prev_fingerprints == new_fingerprints:
        return {
            "bm25_cache_updated": False,
            "bm25_status": "no_change",
            "bm25_total_docs": len(cached_docs),
            "bm25_source_docs": len(clean_docs),
            "bm25_prev_source_docs": len(prev_source_docs),
        }

    merged_docs = unaffected_docs + clean_docs
    preprocess_func = getattr(cached, "preprocess_func", None) or korean_custom_preprocess
    k_value = int(getattr(cached, "k", 5) or 5)

    rebuilt = BM25Retriever.from_documents(
        merged_docs,
        preprocess_func=preprocess_func,
    )
    rebuilt.k = k_value
    _atomic_dump_bm25_cache(rebuilt)
    CACHED_BM25_RETRIEVER = rebuilt

    return {
        "bm25_cache_updated": True,
        "bm25_status": "updated",
        "bm25_total_docs": len(merged_docs),
        "bm25_source_docs": len(clean_docs),
        "bm25_prev_source_docs": len(prev_source_docs),
    }


def get_hybrid_retriever(
    query: str = "",
    collection_name: str = COLLECTION_NAME, 
    vector_weight: float = None, 
    bm25_weight: float = None,
    k: int = 5  # Reranker 부하 감소를 위해 7->5로 최적화
):
    """
    Vector Search (RDS)와 BM25 (Local Cache)를 결합한 하이브리드 리트리버를 반환합니다.
    질문에 기술 용어(에러코드, 모델명) 포함 시 BM25 가중치를 동적으로 상향합니다.
    """
    # 0. 동적 가중치 결정
    # [수정됨] 누락되었던 키워드 추가 (TP630, RB10 등)
    tech_keywords = ["HH", "HI5", "HI6", "UR10", "UR5", "UR3", "DX100", "YRC1000", "TP630", "RB10"]
    
    # [수정됨] 알파벳+숫자 조합(예: Hi5, M0042) 또는 4자리 이상의 연속된 숫자(예: 4107, 38013) 감지
    regex_pattern = r'[A-Za-z]+\d+|\d{4,}'
    
    is_technical = any(kw in query.upper() for kw in tech_keywords) or bool(re.search(regex_pattern, query))
    
    if bm25_weight is None:
        bm25_weight = TECHNICAL_BM25_WEIGHT if is_technical else DEFAULT_BM25_WEIGHT
    if vector_weight is None:
        vector_weight = (1.0 - bm25_weight)
        
    if is_technical:
        print(f"🔍 기술 용어 감지됨 → BM25 가중치 상향 ({bm25_weight})")
    
    print("🚀 하이브리드 리트리버 초기화 중...")
    
    # 1. 벡터 리트리버 설정 (RDS pgvector)
    from app.vectorstore.pgvector_store import get_vector_store
    vector_store = get_vector_store(collection_name=collection_name)
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": k})

    # 2. BM25 리트리버 설정
    bm25_retriever = _load_or_create_bm25_retriever(vector_store)
    bm25_retriever.k = k

    # 3. 앙상블 리트리버 구성
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[vector_weight, bm25_weight]
    )
    
    print(f"✅ 하이브리드 리트리버 준비 완료 (가중치: Vector {vector_weight:.1f}, BM25 {bm25_weight:.1f} | k={k})")
    return ensemble_retriever
