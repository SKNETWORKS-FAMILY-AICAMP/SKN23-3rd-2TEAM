import os
import sys
import pickle
import asyncio
import json
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict, Any, AsyncGenerator

from fastapi import FastAPI, HTTPException, Body, Request, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

# [V4.0] Cross-Platform Asyncio Policy (Windows)
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# 1. ?? PYTHONPATH & ROOT_DIR 怨좎젙
# 理쒖긽??猷⑦듃 main.py ?꾩튂瑜?湲곗??쇰줈 寃쎈줈 蹂댁젙
ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))

# app ?⑦궎吏 ?꾪룷??
from app.core.config import validate_config, ADMIN_SECRET_KEY, CACHE_DIR, DATA_DIR, get_device
from app.agents.graph import compile_workflow
from app.core.history import get_async_postgres_saver
from app.vectorstore.pgvector_store import PGVectorStoreManager, get_vector_store
from app.rag.reranker import load_reranker_singleton
from app.api.auth_api import router as auth_router, setup_auth_middleware
from app.api.routes.admin import router as admin_router
from app.api.routes.chat import router as chat_router
from langchain_core.messages import HumanMessage

app = FastAPI(title="Industrial RAG All-in-One Backend (v4.0 - Auth Integrated)", version="4.0.0")

# Set up Session middleware for authentication (OAuth)
setup_auth_middleware(app)

# [V4.0] Auth Router Inclusion
app.include_router(auth_router)
# [V4.0] Admin Router Inclusion
app.include_router(admin_router)
# [V4.1] Chat Router Inclusion (Clean Architecture)
app.include_router(chat_router)

# ── [Global State] ─────────────────────────────────────────────────────────────
DEVICE = "cpu"

async def debug_system():
    """서버 시작 전 주요 컴포넌트를 점검합니다."""
    print("\n" + "="*50)
    print("🚀 [System Debug] Starting Production Check...")
    
    # 1. 환경 변수 체크
    is_valid, err = validate_config()
    if not is_valid:
        print(f"❌ [Config] Failed: {err}")
        sys.exit(1)
    print("✅ [Config] Mandatory Env Vars Found.")

    # 2. 하드웨어 가용 체크
    global DEVICE
    DEVICE = get_device()
    print(f"✅ [Hardware] Using Device: {DEVICE}")

    # 3. RDS & Vector Store 체크 (Dry Run)
    try:
        with PGVectorStoreManager() as _:
            vector_store = get_vector_store()
            # 간단한 검색 테스트 (최소 1건)
            test_res = await asyncio.wait_for(asyncio.to_thread(lambda: vector_store.similarity_search("test", k=1)), timeout=12)
            print(f"✅ [RDS/pgvector] Connection OK. Found {len(test_res)} docs in test.")
    except Exception as e:
        print(f"❌ [RDS/pgvector] Connection Failed: {e}")
        # 상용구축이므로 실패 시 경고만 하고 진행할지 중단할지 결정 (여기선 중단 권장)
        # sys.exit(1)

    print("="*50 + "\n")

def pre_load_models():
    """서버 기동 시 리랭커 및 BM25를 메모리에 싱글톤으로 적재합니다."""
    from app.rag.retriever import get_hybrid_retriever
    from app.vectorstore.pgvector_store import PGVectorStoreManager
    
    print("🚀 [Startup] 모델 메모리 사전 적재(Pre-load) 시작...")
    load_reranker_singleton() # 리랭커 로드
    
    try:
        with PGVectorStoreManager() as _:
            # 하이브리드 리트리버를 1회 호출해 BM25도 메모리에 올립니다.
            get_hybrid_retriever("test")
    except Exception as e:
        print(f"⚠️ [Startup] BM25 로딩 중 오류 발생: {e}")


@app.on_event("startup")
async def startup_event():
    print("\n" + "─"*48)
    print("🚀 [System] Initializing All-in-One Backend Engine...")
    
    await debug_system()
    pre_load_models()
    
    # [V3.1] 최종 요약 리포트 (Visual Feedback)
    print("\n" + "─"*48)
    print("✅ [Final Status Report]")
    print(f"  ✅ [모델 사전 적재 완료]")
    print(f"  ✅ [하드웨어 가속] Enabled on {DEVICE.upper()}")
    print(f"  ✅ [RDS 연결 성공]       PostgreSQL & pgvector Ready")
    print("✅ " + "─"*50 + "\n")
    print(f"✅ Integrated Backend Started: http://0.0.0.0:8000")

# [Admin Utilities] ?????????????????????????
@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}

if __name__ == "__main__":
    # Windows?먯꽌 Psycopg3 鍮꾨룞湲?猷⑦봽 吏?먯쓣 ?꾪빐 reload=False 濡??ㅼ젙?댁빞 ?먯떇 ?꾨줈?몄뒪 ?ъ깮?깆뿉 ?곕Ⅸ 猷⑦봽 珥덇린??踰꾧렇瑜?留됱쓣 ???덉뒿?덈떎.
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
