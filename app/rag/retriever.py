import os
import pickle
import time
import re
from typing import List
from pathlib import Path

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

def _save_bm25_cache(retriever: BM25Retriever) -> None:
    if not CACHE_DIR.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(BM25_CACHE_PATH, "wb") as f:
        pickle.dump(retriever, f)


def _source_key_variants(source_key: str) -> set[str]:
    key = (source_key or "").strip()
    if not key:
        return set()
    name = Path(key).name
    stem = Path(name).stem
    return {v for v in {key, name, stem, f"{stem}.pdf", f"{stem}.md"} if v}


def _doc_matches_source(doc: Document, source_keys: List[str]) -> bool:
    meta = getattr(doc, "metadata", {}) or {}
    source_values = {
        str(meta.get("source_key", "")).strip(),
        str(meta.get("source_file", "")).strip(),
        Path(str(meta.get("source_file", "")).strip()).name,
        Path(str(meta.get("source", "")).strip()).name,
    }
    source_values = {v for v in source_values if v}

    for sk in source_keys:
        if source_values & _source_key_variants(sk):
            return True
    return False


def _is_doc_active(doc: Document) -> bool:
    meta = getattr(doc, "metadata", {}) or {}
    return str(meta.get("use_yn", "Y")).strip().upper() != "N"

def korean_custom_preprocess(text: str) -> List[str]:
    """
    [?곕떽??? ??? 鈺곌퀣沅?獄??諭?붹묾怨좎깈嚥??紐낅퉸 ?怨몃선/??ъ쁽 ?癒?쑎?꾨뗀諭뜹첎? 筌띲끉臾??? ??낅뮉 ?袁⑷맒 獄쎻뫗?
    ?? "M0042揶쎛" -> "M0042 揶쎛" 嚥??브쑬???????덈즲嚥??諭?붹묾怨좎깈 ?類ㅼ젫 ???袁⑸선?怨뚮┛ 疫꿸퀣? ?브쑬??
    """
    if not isinstance(text, str):
        return []
    # ???, ?怨론? ??ъ쁽????뽰뇚??筌뤴뫀諭??諭?붹묾怨좎깈???⑤벉媛??곗쨮 燁살꼹??
    text = re.sub(r'[^\\uac00-\\ud7a3A-Za-z0-9]', ' ', text)
    return text.split()

def _load_or_create_bm25_retriever(vector_store, force_refresh: bool = False):
    """
    BM25 ?귐뗫뱜?귐됱쒔??筌롫뗀?덄뵳?Singleton) -> 筌?Ŋ?????뵬 -> RDS ??뽰몵嚥?嚥≪뮆諭??몃빍??
    """
    global CACHED_BM25_RETRIEVER
    
    if CACHED_BM25_RETRIEVER is not None and not force_refresh:
        return CACHED_BM25_RETRIEVER

    if not CACHE_DIR.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if BM25_CACHE_PATH.exists() and not force_refresh:
        print(f"📦 BM25 인덱스 캐시 로드 중 (File): {BM25_CACHE_PATH}")
        with open(BM25_CACHE_PATH, "rb") as f:
            CACHED_BM25_RETRIEVER = pickle.load(f)
            return CACHED_BM25_RETRIEVER
            
    if force_refresh and BM25_CACHE_PATH.exists():
        try:
            BM25_CACHE_PATH.unlink()
            print("🗑️ 기존 BM25 캐시 파일을 삭제했습니다.")
        except Exception as e:
            pass
    
    print("⚠️ BM25 캐시가 없습니다. RDS에서 모든 문서를 로드하여 인덱스를 생성합니다.")
    start_time = time.time()
    
    try:
        # RDS?癒?퐣 筌뤴뫀諭??얜챷苑?揶쎛?紐꾩궎疫?(?袁⑥쎗 嚥≪뮆諭?
        # 雅뚯눘?? ???얜챷???野꺜??깆뵠 ??? ?袁⑥퓢??筌뤴뫀??癒?퐣 ?븍뜆釉?類λ막 野껋럩??vector_store.get() ????뽰뒠 亦낅슣??
        all_docs = vector_store.similarity_search("", k=100000)
        all_docs = [doc for doc in all_docs if _is_doc_active(doc)]
        print(f"[BM25] Loaded {len(all_docs)} docs. Building index...")
        
        # [??륁젟?? ?뚣끉??? ?袁⑹퓗????λ땾(korean_custom_preprocess) ?怨몄뒠
        bm25_retriever = BM25Retriever.from_documents(
            all_docs,
            preprocess_func=korean_custom_preprocess
        )
        bm25_retriever.k = 5
        
        # 筌?Ŋ???????
        with open(BM25_CACHE_PATH, "wb") as f:
            pickle.dump(bm25_retriever, f)
            
        CACHED_BM25_RETRIEVER = bm25_retriever
        end_time = time.time()
        print(f"[BM25] Index build + cache save complete ({end_time - start_time:.2f}s)")
        return CACHED_BM25_RETRIEVER
        
    except Exception as e:
        print(f"[BM25] Index build failed: {e}")
        raise e

def get_hybrid_retriever(
    query: str = "",
    collection_name: str = COLLECTION_NAME, 
    vector_weight: float = None, 
    bm25_weight: float = None,
    k: int = 5  # default per-retriever candidate count
):
    """
    Vector Search (RDS)?? BM25 (Local Cache)??野껉퀬鍮????륁뵠?됰슢????귐뗫뱜?귐됱쒔??獄쏆꼹???몃빍??
    筌욌뜄揆??疫꿸퀣????밸선(?癒?쑎?꾨뗀諭? 筌뤴뫀?쏙쭗? ??釉???BM25 揶쎛餓λ쵐?귞몴???덉읅??곗쨮 ?怨밸샨??몃빍??
    """
    # 0. ??덉읅 揶쎛餓λ쵐??野껉퀣??
    # [??륁젟?? ?袁⑥뵭??뤿?????쇱뜖???곕떽? (TP630, RB10 ??
    tech_keywords = ["HH", "HI5", "HI6", "UR10", "UR5", "UR3", "DX100", "YRC1000", "TP630", "RB10"]
    
    # [??륁젟?? ??곕솁甕???ъ쁽 鈺곌퀬鍮(?? Hi5, M0042) ?癒?뮉 4?癒?봺 ??곴맒???怨쀫꺗????ъ쁽(?? 4107, 38013) 揶쏅Ŋ?
    regex_pattern = r'[A-Za-z]+\d+|\d{4,}'
    
    is_technical = any(kw in query.upper() for kw in tech_keywords) or bool(re.search(regex_pattern, query))
    
    if bm25_weight is None:
        bm25_weight = TECHNICAL_BM25_WEIGHT if is_technical else DEFAULT_BM25_WEIGHT
    if vector_weight is None:
        vector_weight = (1.0 - bm25_weight)
        
    if is_technical:
        print(f"[Retriever] Technical query detected -> increase BM25 weight ({bm25_weight})")
    
    print("[Retriever] Initializing hybrid retriever...")
    # 1. 甕겸돧苑??귐뗫뱜?귐됱쒔 ??쇱젟 (RDS pgvector)
    from app.vectorstore.pgvector_store import get_vector_store
    vector_store = get_vector_store(collection_name=collection_name)
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": k})

    # 2. BM25 ?귐뗫뱜?귐됱쒔 ??쇱젟
    bm25_retriever = _load_or_create_bm25_retriever(vector_store)
    bm25_retriever.k = k

    # 3. ??덇맒???귐뗫뱜?귐됱쒔 ?닌딄쉐
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[vector_weight, bm25_weight]
    )
    
    print(f"[Retriever] Hybrid retriever ready (weights: Vector {vector_weight:.1f}, BM25 {bm25_weight:.1f} | k={k})")
    return ensemble_retriever

def refresh_bm25_index():
    """관리자 PDF 업로드 후 BM25 캐시 및 메모리 상주 객체를 강제로 초기화 및 재구축합니다."""
    global CACHED_BM25_RETRIEVER
    print("🔄 BM25 인덱스 전격 강제 초기화 (Refresh) 시작...")
    CACHED_BM25_RETRIEVER = None
    from app.vectorstore.pgvector_store import get_vector_store
    from app.vectorstore.pgvector_store import PGVectorStoreManager
    
    with PGVectorStoreManager() as _:
        vs = get_vector_store(collection_name=COLLECTION_NAME)
        _load_or_create_bm25_retriever(vs, force_refresh=True)
    
    print("✅ BM25 인덱스 글로벌 리프레시 완료!")

def update_bm25_cache_for_uploaded_source(source_keys: List[str]):
    """
    관리자 PDF 업로드 시 새로 추가되거나 삭제된 문서만 캐시에 반영하여 
    RDS 전체 로드를 방지하는 준-증분 업데이트 로직
    """
    global CACHED_BM25_RETRIEVER
    if not source_keys:
        return
        
    print(f"⚡ [BM25] 준-증분 업데이트 시도: {source_keys}")
    
    try:
        from app.vectorstore.pgvector_store import get_vector_store
        vs = get_vector_store(collection_name=COLLECTION_NAME)
        
        # 1. 파일이 존재하는지 확인하고 기존 캐시 로드 시도
        if CACHED_BM25_RETRIEVER is None:
            # 캐시가 아예 메모리에 없으면(처음 올렸을 때 등) load_or_create 를 탄다.
            # 이 경우 전체 갱신이 일어날 수도 있지만, 이미 존재하는 파일이 없을 경우 대비
            if not BM25_CACHE_PATH.exists():
                 print("⚠️ 기존 BM25 캐시가 없어 기본 리프레시를 수행합니다.")
                 refresh_bm25_index()
                 return
                 
            with open(BM25_CACHE_PATH, "rb") as f:
                 CACHED_BM25_RETRIEVER = pickle.load(f)
                 
        if CACHED_BM25_RETRIEVER is None:
             refresh_bm25_index()
             return
             
        # 기존 문서 보존하면서 삭제된 소스를 필터링
        print(f"🗑️ [BM25] 기존 인덱스에서 삭제 및 업데이트 대상 필터링 중...")
        filtered_docs = []
        for d in CACHED_BM25_RETRIEVER.docs:
            if not _is_doc_active(d):
               continue
            if not _doc_matches_source(d, source_keys):
               filtered_docs.append(d)
               
        # 새로운 문서 가져오기 (RDS 활용)
        print(f"📥 [BM25] RDS에서 새로운({len(source_keys)}개 소스) 청크 로딩 중...")
        new_docs = []
        # source_key 로 메타데이터 필터링하여 가져온다. langchain-pgvector의 search 방식 활용
        # 모든 청크를 가져오려면 비어있는 query를 사용하고 filter 적용
        for source_key in source_keys:
             variants = _source_key_variants(source_key)
             source_docs = []
             for candidate in variants:
                  docs = vs.similarity_search(
                      "",
                      k=10000,
                      filter={"source_file": candidate}
                  )
                  docs = [doc for doc in docs if _is_doc_active(doc)]
                  source_docs.extend(docs)

             # source_file 매칭이 실패한 경우를 대비해 source_key도 시도
             if not source_docs:
                  docs = vs.similarity_search(
                      "",
                      k=10000,
                      filter={"source_key": source_key}
                  )
                  docs = [doc for doc in docs if _is_doc_active(doc)]
                  source_docs.extend(docs)

             new_docs.extend(source_docs)

        # 중복 문서 제거 (다중 variant filter 조회 시 중복 방지)
        if new_docs:
            deduped = []
            seen = set()
            for d in new_docs:
                meta = getattr(d, "metadata", {}) or {}
                sig = (d.page_content, tuple(sorted((str(k), str(v)) for k, v in meta.items())))
                if sig in seen:
                    continue
                seen.add(sig)
                deduped.append(d)
            new_docs = deduped

        # 합친 후 새로운 인덱스 생성
        combined_docs = filtered_docs + new_docs
        print(f"🔄 [BM25] 병합 완료: 총 {len(combined_docs)} 문서를 사용하여 BM25 재구축 연산 시작...")
        
        start_time = time.time()
        new_bm25 = BM25Retriever.from_documents(
            combined_docs,
            preprocess_func=korean_custom_preprocess
        )
        new_bm25.k = CACHED_BM25_RETRIEVER.k
        
        # 캐싱 및 덮어쓰기
        _save_bm25_cache(new_bm25)
            
        CACHED_BM25_RETRIEVER = new_bm25
        end_time = time.time()
        print(f"✅ [BM25] 준-증분 업데이트 완료 ({end_time - start_time:.2f}초 소요)")
        
    except Exception as e:
        print(f"❌ [BM25] 준-증분 업데이트 오류: {e}")
        print("Fallback으로 전체 초기화를 진행합니다.")
        refresh_bm25_index()


def remove_sources_from_bm25_cache(source_keys: List[str]) -> dict:
    """
    비활성화 처리한 source_key의 문서를 BM25 캐시에서 물리적으로 제거합니다.
    """
    global CACHED_BM25_RETRIEVER

    keys = [k.strip() for k in (source_keys or []) if isinstance(k, str) and k.strip()]
    if not keys:
        return {
            "bm25_cache_updated": False,
            "bm25_status": "skipped_empty_source_keys",
            "bm25_removed_docs": 0,
            "bm25_total_docs": None,
        }

    if CACHED_BM25_RETRIEVER is None:
        if not BM25_CACHE_PATH.exists():
            return {
                "bm25_cache_updated": False,
                "bm25_status": "skipped_cache_not_found",
                "bm25_removed_docs": 0,
                "bm25_total_docs": None,
            }
        with open(BM25_CACHE_PATH, "rb") as f:
            CACHED_BM25_RETRIEVER = pickle.load(f)

    current_docs = list(getattr(CACHED_BM25_RETRIEVER, "docs", []) or [])
    if not current_docs:
        return {
            "bm25_cache_updated": False,
            "bm25_status": "skipped_empty_cache",
            "bm25_removed_docs": 0,
            "bm25_total_docs": 0,
        }

    kept_docs = []
    removed_count = 0
    for doc in current_docs:
        if _doc_matches_source(doc, keys):
            removed_count += 1
            continue
        kept_docs.append(doc)

    if removed_count == 0:
        return {
            "bm25_cache_updated": False,
            "bm25_status": "no_match",
            "bm25_removed_docs": 0,
            "bm25_total_docs": len(current_docs),
        }

    preprocess_func = getattr(CACHED_BM25_RETRIEVER, "preprocess_func", None) or korean_custom_preprocess
    k_value = int(getattr(CACHED_BM25_RETRIEVER, "k", 5) or 5)
    rebuilt = BM25Retriever.from_documents(kept_docs, preprocess_func=preprocess_func)
    rebuilt.k = k_value

    _save_bm25_cache(rebuilt)
    CACHED_BM25_RETRIEVER = rebuilt

    return {
        "bm25_cache_updated": True,
        "bm25_status": "removed",
        "bm25_removed_docs": removed_count,
        "bm25_total_docs": len(kept_docs),
    }
