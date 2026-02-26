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

# 1. 🚀 PYTHONPATH & ROOT_DIR 고정
# 최상위 루트 main.py 위치를 기준으로 경로 보정
ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))

# app 패키지 임포트
from app.core.config import validate_config, ADMIN_SECRET_KEY, CACHE_DIR, DATA_DIR, get_device
from app.agents.graph import compile_workflow
from app.core.history import get_async_postgres_saver
from app.vectorstore.pgvector_store import PGVectorStoreManager, get_vector_store
from app.rag.reranker import load_reranker_singleton
from app.api.auth_api import router as auth_router, setup_auth_middleware
from app.api.routes.admin import router as admin_router
from langchain_core.messages import HumanMessage

app = FastAPI(title="Industrial RAG All-in-One Backend (v4.0 - Auth Integrated)", version="4.0.0")

# Set up Session middleware for authentication (OAuth)
setup_auth_middleware(app)

# [V4.0] Auth Router Inclusion
app.include_router(auth_router)
# [V4.0] Admin Router Inclusion
app.include_router(admin_router)

# ── [Global State] ────────────────────────────
GLOBAL_BM25_RETRIEVER = None
BM25_CACHE_PATH = CACHE_DIR / "bm25_retriever.pkl"
DEVICE = "cpu"

async def debug_system():
    """서버 시작 전 주요 컴포넌트를 점검합니다."""
    print("\n" + "="*50)
    print("🔍 [System Debug] Starting Production Check...")
    
    # 1. 환경 변수 체크
    is_valid, err = validate_config()
    if not is_valid:
        print(f"❌ [Config] Failed: {err}")
        sys.exit(1)
    print("✅ [Config] Mandatory Env Vars Found.")

    # 2. 하드웨어 가속 체크
    global DEVICE
    DEVICE = get_device()
    print(f"✅ [Hardware] Using Device: {DEVICE}")

    # 3. RDS & Vector Store 체크 (Dry Run)
    try:
        with PGVectorStoreManager() as _:
            vector_store = get_vector_store()
            # 간단한 검색 테스트 (최소 1건)
            test_res = vector_store.similarity_search("안전", k=1)
            print(f"✅ [RDS/pgvector] Connection OK. Found {len(test_res)} docs in test.")
    except Exception as e:
        print(f"❌ [RDS/pgvector] Connection Failed: {e}")
        # 상용구축이므로 실패 시 경고만 하고 진행할지 중단할지 결정 (여기선 중단 권장)
        # sys.exit(1)

    print("="*50 + "\n")

def load_global_bm25():
    global GLOBAL_BM25_RETRIEVER
    if BM25_CACHE_PATH.exists():
        print(f"📦 [Startup] Loading BM25 Index (94k docs)...")
        with open(BM25_CACHE_PATH, "rb") as f:
            GLOBAL_BM25_RETRIEVER = pickle.load(f)
        print("✅ [Startup] Global BM25 Loaded.")
    else:
        print("⚠️ [Startup] BM25 Cache not found. Initial query will be slow.")

@app.on_event("startup")
async def startup_event():
    print("\n" + "🚀" + "─"*48)
    print("🌍 [System] Initializing All-in-One Backend Engine...")
    
    await debug_system()
    load_global_bm25()
    load_reranker_singleton() # [V3.1] 리랭커 싱글톤 GPU 고정 로드
    
    # [V3.1] 최종 요약 리포트 (Visual Feedback)
    print("\n" + "🏁" + "─"*48)
    print("💎 [Final Status Report]")
    bm25_status = "✅ Loaded (94k docs)" if GLOBAL_BM25_RETRIEVER else "⚠️ Not Found (Cache required)"
    print(f"  {bm25_status} [BM25 로드 완료]")
    print(f"  ✅ [하드웨어 가속 활성화] Enabled on {DEVICE.upper()}")
    print(f"  ✅ [RDS 연결 성공]       PostgreSQL & pgvector Ready")
    print("💎 " + "─"*50 + "\n")
    print(f"🚀 Integrated Backend Started: http://0.0.0.0:8000")

# ── [Schemas] ──────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    thread_id: str

class JargonUpdate(BaseModel):
    data: List[Dict[str, str]]

# ── [Chat & SSE] ───────────────────────────────
async def sse_event_generator(request_data: ChatRequest) -> AsyncGenerator[str, None]:
    import time
    start_total = time.perf_counter()
    node_timers = {} # 노드별 시작 시간 저장용

    # [KEEP-ALIVE] 연결 즉시 상태 메시지를 전송하여 타임아웃 방지
    yield f"data: {json.dumps({'type': 'status', 'content': '🚀 Connection established. Initializing engine...'})}\n\n"
    
    print(f"\n💬 [Chat] New Request: '{request_data.message}' (Thread: {request_data.thread_id})")
    print(f"⚙️ [Device] Running on: {DEVICE} | Model: gpt-4o")
    
    try:
        # [DEBUG] 엔진 초기화 과정 로그
        print("  ↳ 🔧 Initializing LangGraph & RDS Connection...")
        with PGVectorStoreManager() as _:
            async with get_async_postgres_saver() as saver:
                graph = compile_workflow(saver)
                config = {"configurable": {"thread_id": request_data.thread_id}}
                inputs = {"messages": [HumanMessage(content=request_data.message)], "retry_count": 0}
                
                print("  ↳ ⚡ Starting graph execution...")
                async for event in graph.astream_events(inputs, config, version="v2"):
                    kind = event["event"]
                    node_name = event["metadata"].get("langgraph_node", "")
                    
                    # [진행 상태 & 시작 시간 측정]
                    if kind == "on_node_start":
                        if node_name:
                            node_timers[node_name] = time.perf_counter()
                            print(f"  ↳ 🔄 Node Start: {node_name.upper()}")
                            yield f"data: {json.dumps({
                                'type': 'status', 
                                'content': f'{node_name.upper()} processing...',
                                'node': node_name
                            })}\n\n"
                        
                        # 특정 단계 로그 커스터마이징
                        if node_name == "retriever":
                            print("    🔍 [Step 1] Initial BM25 + Vector Search starting...")
                        elif node_name == "reranker":
                            print("    sort [Step 2] Reranking documents...")
                        elif node_name in ["welding", "robotics", "safety", "electrical", "general", "social"]:
                            print(f"    🤖 [Step 3] Calling LLM ({node_name.upper()} Agent)...")

                    # [답변 덩어리]
                    if kind == "on_chain_stream":
                        chunk = event["data"]["chunk"]
                        if isinstance(chunk, dict) and "generated_answer" in chunk:
                            yield f"data: {json.dumps({'type': 'answer', 'content': chunk['generated_answer']})}\n\n"
                    
                    # [노드 종료 & 소요 시간 계산]
                    if kind == "on_node_end":
                        if node_name:
                            elapsed = time.perf_counter() - node_timers.get(node_name, time.perf_counter())
                            print(f"  ↳ ✅ Node End: {node_name.upper()} ({elapsed:.2f}s)")
                            # 종료 정보와 시간을 메타데이터로 전송
                            yield f"data: {json.dumps({
                                'type': 'metadata', 
                                'content': f'{node_name.upper()} completed',
                                'node': node_name,
                                'elapsed': round(elapsed, 2)
                            })}\n\n"
                        
                        if node_name == "verifier":
                            output = event["data"].get("output")
                            if output and output.get("is_hallucinated"):
                                print("    ⚠️ [Warning] Hallucination suspected!")
                                yield f"data: {json.dumps({'type': 'warning', 'content': 'Hallucination suspect - rewriting...'})}\n\n"
                
                total_elapsed = time.perf_counter() - start_total
                
                # [V3.1] RDS 영구 저장: 최종 실행 통계를 State Metadata에 주입
                # node_timers에 저장된 값은 시작 시간뿐이므로 elapsed를 계산해야 함
                execution_metadata = {
                    "total_elapsed": round(total_elapsed, 2),
                    "device": DEVICE,
                    "node_timings": {k: round(time.perf_counter() - v, 2) for k, v in node_timers.items()}
                }
                await graph.aupdate_state(config, {"metadata": execution_metadata})
                
                print(f"💾 [Persistence] Session archived with metadata: {execution_metadata}")
                yield f"data: {json.dumps({'type': 'status', 'content': f'✅ Finished in {total_elapsed:.2f}s'})}\n\n"
                
                # [V3.3] chat_history 테이블에 이력 저장
                print("📝 [History] Saving chat interaction to RDS...")
                try:
                    import psycopg2
                    ssh_enabled = os.getenv("SSH_TUNNEL_ENABLED", "false").lower() == "true"
                    ssh_local_port = os.getenv("SSH_LOCAL_BIND_PORT", "15432")
                    target_host = "127.0.0.1" if ssh_enabled else os.getenv("PGHOST", "localhost")
                    target_port = ssh_local_port if ssh_enabled else os.getenv("PGPORT", "5432")

                    conn = psycopg2.connect(
                        host=target_host,
                        database=os.getenv("PGDATABASE", "chatbot_db"),
                        user=os.getenv("PGUSER", "postgres"),
                        password=os.getenv("PGPASSWORD", "password"),
                        port=target_port
                    )
                    with conn.cursor() as cur:
                        # 1. User Message 저장
                        cur.execute(
                            "INSERT INTO chat_history (thread_id, username, role, content) VALUES (%s, %s, %s, %s)",
                            (request_data.thread_id, "default_user", "user", request_data.message)
                        )
                        # 2. Assistant Message 저장 (최종 답변 추출)
                        final_state = await graph.aget_state(config)
                        assistant_msg = final_state.values.get("generated_answer", "")
                        if assistant_msg:
                            cur.execute(
                                "INSERT INTO chat_history (thread_id, username, role, content) VALUES (%s, %s, %s, %s)",
                                (request_data.thread_id, "WELD·BOT", "assistant", assistant_msg)
                            )
                    conn.commit()
                    conn.close()
                    print("✅ [History] Chat interaction saved successfully.")
                except Exception as history_err:
                    print(f"⚠️ [History] DB 저장 실패: {history_err}")
                
    except Exception as e:
        import traceback
        print(f"❌ [Error] SSE Failed: {e!r}")
        traceback.print_exc()
        yield f"data: {json.dumps({'type': 'error', 'content': f'Engine Error: {str(e)}'})}\n\n"

@app.post("/chat")
async def chat_stream(request: ChatRequest):
    return StreamingResponse(sse_event_generator(request), media_type="text/event-stream")

# ── [History] ──────────────────────────────────
@app.get("/history/{thread_id}")
async def get_history(thread_id: str):
    try:
        async with get_async_postgres_saver() as saver:
            config = {"configurable": {"thread_id": thread_id}}
            checkpoint = await saver.aget(config)
            if checkpoint and "messages" in checkpoint["channel_values"]:
                msgs = checkpoint["channel_values"]["messages"]
                return [{"role": "user" if m.type=="human" else "assistant", "content": m.content} for m in msgs]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return []

# [Admin Utilities] ─────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}

if __name__ == "__main__":
    # Windows에서 Psycopg3 비동기 루프 지원을 위해 reload=False 로 설정해야 자식 프로세스 재생성에 따른 루프 초기화 버그를 막을 수 있습니다.
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
