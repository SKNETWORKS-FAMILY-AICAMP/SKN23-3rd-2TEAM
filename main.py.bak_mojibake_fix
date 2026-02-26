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
from langchain_core.messages import HumanMessage

app = FastAPI(title="Industrial RAG All-in-One Backend (v4.0 - Auth Integrated)", version="4.0.0")

# Set up Session middleware for authentication (OAuth)
setup_auth_middleware(app)

# [V4.0] Auth Router Inclusion
app.include_router(auth_router)
# [V4.0] Admin Router Inclusion
app.include_router(admin_router)

# ?? [Global State] ????????????????????????????
GLOBAL_BM25_RETRIEVER = None
BM25_CACHE_PATH = CACHE_DIR / "bm25_retriever.pkl"
DEVICE = "cpu"

async def debug_system():
    """?쒕쾭 ?쒖옉 ??二쇱슂 而댄룷?뚰듃瑜??먭??⑸땲??"""
    print("\n" + "="*50)
    print("?뵇 [System Debug] Starting Production Check...")
    
    # 1. ?섍꼍 蹂??泥댄겕
    is_valid, err = validate_config()
    if not is_valid:
        print(f"??[Config] Failed: {err}")
        sys.exit(1)
    print("??[Config] Mandatory Env Vars Found.")

    # 2. ?섎뱶?⑥뼱 媛??泥댄겕
    global DEVICE
    DEVICE = get_device()
    print(f"??[Hardware] Using Device: {DEVICE}")

    # 3. RDS & Vector Store 泥댄겕 (Dry Run)
    try:
        with PGVectorStoreManager() as _:
            vector_store = get_vector_store()
            # 媛꾨떒??寃???뚯뒪??(理쒖냼 1嫄?
            test_res = await asyncio.wait_for(asyncio.to_thread(lambda: vector_store.similarity_search("test", k=1)), timeout=12)
            print(f"??[RDS/pgvector] Connection OK. Found {len(test_res)} docs in test.")
    except Exception as e:
        print(f"??[RDS/pgvector] Connection Failed: {e}")
        # ?곸슜援ъ텞?대?濡??ㅽ뙣 ??寃쎄퀬留??섍퀬 吏꾪뻾?좎? 以묐떒?좎? 寃곗젙 (?ш린??以묐떒 沅뚯옣)
        # sys.exit(1)

    print("="*50 + "\n")

def load_global_bm25():
    global GLOBAL_BM25_RETRIEVER
    if BM25_CACHE_PATH.exists():
        print(f"?벀 [Startup] Loading BM25 Index (94k docs)...")
        with open(BM25_CACHE_PATH, "rb") as f:
            GLOBAL_BM25_RETRIEVER = pickle.load(f)
        print("??[Startup] Global BM25 Loaded.")
    else:
        print("?좑툘 [Startup] BM25 Cache not found. Initial query will be slow.")

@app.on_event("startup")
async def startup_event():
    print("\n" + "??" + "?"*48)
    print("?뙇 [System] Initializing All-in-One Backend Engine...")
    
    await debug_system()
    load_global_bm25()
    load_reranker_singleton() # [V3.1] 由щ옲而??깃???GPU 怨좎젙 濡쒕뱶
    
    # [V3.1] 理쒖쥌 ?붿빟 由ы룷??(Visual Feedback)
    print("\n" + "?뢾" + "?"*48)
    print("?뭿 [Final Status Report]")
    bm25_status = "??Loaded (94k docs)" if GLOBAL_BM25_RETRIEVER else "?좑툘 Not Found (Cache required)"
    print(f"  {bm25_status} [BM25 濡쒕뱶 ?꾨즺]")
    print(f"  ??[?섎뱶?⑥뼱 媛???쒖꽦?? Enabled on {DEVICE.upper()}")
    print(f"  ??[RDS ?곌껐 ?깃났]       PostgreSQL & pgvector Ready")
    print("?뭿 " + "?"*50 + "\n")
    print(f"?? Integrated Backend Started: http://0.0.0.0:8000")

# ?? [Schemas] ??????????????????????????????????
class ChatRequest(BaseModel):
    message: str
    thread_id: str

class JargonUpdate(BaseModel):
    data: List[Dict[str, str]]

# ?? [Chat & SSE] ???????????????????????????????
async def sse_event_generator(request_data: ChatRequest) -> AsyncGenerator[str, None]:
    import time
    start_total = time.perf_counter()
    node_timers = {} # ?몃뱶蹂??쒖옉 ?쒓컙 ??μ슜

    # [KEEP-ALIVE] ?곌껐 利됱떆 ?곹깭 硫붿떆吏瑜??꾩넚?섏뿬 ??꾩븘??諛⑹?
    yield f"data: {json.dumps({'type': 'status', 'content': '?? Connection established. Initializing engine...'})}\n\n"
    
    print(f"\n?뮠 [Chat] New Request: '{request_data.message}' (Thread: {request_data.thread_id})")
    print(f"?숋툘 [Device] Running on: {DEVICE} | Model: gpt-4o")
    
    try:
        # [DEBUG] ?붿쭊 珥덇린??怨쇱젙 濡쒓렇
        print("  ???뵩 Initializing LangGraph & RDS Connection...")
        with PGVectorStoreManager() as _:
            async with get_async_postgres_saver() as saver:
                graph = compile_workflow(saver)
                config = {"configurable": {"thread_id": request_data.thread_id}}
                inputs = {"messages": [HumanMessage(content=request_data.message)], "retry_count": 0}
                
                print("  ????Starting graph execution...")
                async for event in graph.astream_events(inputs, config, version="v2"):
                    kind = event["event"]
                    node_name = event["metadata"].get("langgraph_node", "")
                    
                    # [吏꾪뻾 ?곹깭 & ?쒖옉 ?쒓컙 痢≪젙]
                    if kind == "on_node_start":
                        if node_name:
                            node_timers[node_name] = time.perf_counter()
                            print(f"  ???봽 Node Start: {node_name.upper()}")
                            yield f"data: {json.dumps({
                                'type': 'status', 
                                'content': f'{node_name.upper()} processing...',
                                'node': node_name
                            })}\n\n"
                        
                        # ?뱀젙 ?④퀎 濡쒓렇 而ㅼ뒪?곕쭏?댁쭠
                        if node_name == "retriever":
                            print("    ?뵇 [Step 1] Initial BM25 + Vector Search starting...")
                        elif node_name == "reranker":
                            print("    sort [Step 2] Reranking documents...")
                        elif node_name in ["welding", "robotics", "safety", "electrical", "general", "social"]:
                            print(f"    ?쨼 [Step 3] Calling LLM ({node_name.upper()} Agent)...")

                    # [?듬? ?⑹뼱由?
                    if kind == "on_chain_stream":
                        chunk = event["data"]["chunk"]
                        if isinstance(chunk, dict) and "generated_answer" in chunk:
                            yield f"data: {json.dumps({'type': 'answer', 'content': chunk['generated_answer']})}\n\n"
                    
                    # [?몃뱶 醫낅즺 & ?뚯슂 ?쒓컙 怨꾩궛]
                    if kind == "on_node_end":
                        if node_name:
                            elapsed = time.perf_counter() - node_timers.get(node_name, time.perf_counter())
                            print(f"  ????Node End: {node_name.upper()} ({elapsed:.2f}s)")
                            # 醫낅즺 ?뺣낫? ?쒓컙??硫뷀??곗씠?곕줈 ?꾩넚
                            yield f"data: {json.dumps({
                                'type': 'metadata', 
                                'content': f'{node_name.upper()} completed',
                                'node': node_name,
                                'elapsed': round(elapsed, 2)
                            })}\n\n"
                        
                        if node_name == "verifier":
                            output = event["data"].get("output")
                            if output and output.get("is_hallucinated"):
                                print("    ?좑툘 [Warning] Hallucination suspected!")
                                yield f"data: {json.dumps({'type': 'warning', 'content': 'Hallucination suspect - rewriting...'})}\n\n"
                
                total_elapsed = time.perf_counter() - start_total
                
                # [V3.1] RDS ?곴뎄 ??? 理쒖쥌 ?ㅽ뻾 ?듦퀎瑜?State Metadata??二쇱엯
                # node_timers????λ맂 媛믪? ?쒖옉 ?쒓컙肉먯씠誘濡?elapsed瑜?怨꾩궛?댁빞 ??
                execution_metadata = {
                    "total_elapsed": round(total_elapsed, 2),
                    "device": DEVICE,
                    "node_timings": {k: round(time.perf_counter() - v, 2) for k, v in node_timers.items()}
                }
                await graph.aupdate_state(config, {"metadata": execution_metadata})
                
                print(f"?뮶 [Persistence] Session archived with metadata: {execution_metadata}")
                yield f"data: {json.dumps({'type': 'status', 'content': f'??Finished in {total_elapsed:.2f}s'})}\n\n"
                
                # [V3.3] chat_history ?뚯씠釉붿뿉 ?대젰 ???
                print("?뱷 [History] Saving chat interaction to RDS...")
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
                        # 1. User Message ???
                        cur.execute(
                            "INSERT INTO chat_history (thread_id, username, role, content) VALUES (%s, %s, %s, %s)",
                            (request_data.thread_id, "default_user", "user", request_data.message)
                        )
                        # 2. Assistant Message ???(理쒖쥌 ?듬? 異붿텧)
                        final_state = await graph.aget_state(config)
                        assistant_msg = final_state.values.get("generated_answer", "")
                        if assistant_msg:
                            cur.execute(
                                "INSERT INTO chat_history (thread_id, username, role, content) VALUES (%s, %s, %s, %s)",
                                (request_data.thread_id, "WELD쨌BOT", "assistant", assistant_msg)
                            )
                    conn.commit()
                    conn.close()
                    print("??[History] Chat interaction saved successfully.")
                except Exception as history_err:
                    print(f"?좑툘 [History] DB ????ㅽ뙣: {history_err}")
                
    except Exception as e:
        import traceback
        print(f"??[Error] SSE Failed: {e!r}")
        traceback.print_exc()
        yield f"data: {json.dumps({'type': 'error', 'content': f'Engine Error: {str(e)}'})}\n\n"

@app.post("/chat")
async def chat_stream(request: ChatRequest):
    return StreamingResponse(sse_event_generator(request), media_type="text/event-stream")

# ?? [History] ??????????????????????????????????
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

# [Admin Utilities] ?????????????????????????
@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}

if __name__ == "__main__":
    # Windows?먯꽌 Psycopg3 鍮꾨룞湲?猷⑦봽 吏?먯쓣 ?꾪빐 reload=False 濡??ㅼ젙?댁빞 ?먯떇 ?꾨줈?몄뒪 ?ъ깮?깆뿉 ?곕Ⅸ 猷⑦봽 珥덇린??踰꾧렇瑜?留됱쓣 ???덉뒿?덈떎.
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
