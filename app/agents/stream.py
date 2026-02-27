# ============================================================
# [비동기 스트리밍 실행기] app/agents/stream.py
# ============================================================
# 역할: LangGraph의 astream_events를 활용해 LLM이 생성하는
#       텍스트 청크를 실시간으로 Yield합니다.
#       FastAPI의 SSE (StreamingResponse) 또는 WebSocket에 직접 연결할 수 있습니다.
#
# 필수 Import:
#   from langgraph.graph import CompiledGraph  -- 컴파일된 그래프 인스턴스
#   from langchain_core.messages import HumanMessage
# ============================================================
import asyncio
from typing import AsyncGenerator
from langchain_core.messages import HumanMessage
from app.agents.graph import compile_workflow
from app.schemas.state import GraphState

async def stream_chat_response(
    user_message: str,
    thread_id: str,
    user_id: str = "Unknown",
) -> AsyncGenerator[str, None]:
    """
    사용자 메시지를 LangGraph에 전달하고, LLM이 생성하는 텍스트 청크를
    실시간으로 Yield하는 비동기 제너레이터입니다.

    FastAPI StreamingResponse / SSE / WebSocket에 직접 연결하세요.

    Args:
        user_message (str): 사용자의 질문 텍스트
        thread_id (str):    세션 식별자 (사용자별 대화 이력 분리)

    Yields:
        str: LLM이 생성하는 텍스트 청크 (토큰 단위)

    [사용 예시 - FastAPI SSE]
        @app.get("/chat/stream")
        async def chat_stream(q: str, thread_id: str):
            return StreamingResponse(
                stream_chat_response(q, thread_id),
                media_type="text/event-stream"
            )
    """
    config = {
        "configurable": {
            "thread_id": thread_id
            # thread_id: MemorySaver가 이 값으로 대화 이력을 분리합니다.
            # 동일한 thread_id = 같은 대화방 (멀티턴 유지)
        }
    }

    # [V4.1] DB 또는 메모리 저장소에서 과거 메시지 이력을 가져온 뒤 최근 3턴(6개 메시지)으로 트리밍(Trimming)
    # LangGraph의 StateGraph는 체크포인트에 저장된 `messages` 리스트에 새 메시지를 append합니다.
    # 히스토리가 무한정 길어지는 것을 방지하기 위해 별도의 memory trimming 노드를 사용할 수도 있으나,
    # 여기서는 프롬프트 주입 전 rewriter와 specialist 노드 내부에서 `messages[-6:]` 형태로 자르거나, 
    # 혹은 Saver에서 불러온 값을 명시적으로 잘라 줄 수 있습니다.
    #
    # 최적화: `app_graph.ainvoke`나 `astream_events` 호출 전,
    # 체크포인터(DB)에서 기존 히스토리를 aget()으로 불러와서 너무 길면
    # `messages` 리스트 앞부분을 잘라내는 로직을 추가합니다.
    
    from app.core.history import get_async_postgres_saver
    from langchain_core.messages import RemoveMessage
    
    # ── 히스토리 트리밍 로직 및 그래프 초기화 ──
    async with get_async_postgres_saver() as saver:
        app_graph = compile_workflow(saver)
        
        checkpoint = await saver.aget(config)
        if checkpoint and "messages" in checkpoint["channel_values"]:
            old_messages = checkpoint["channel_values"]["messages"]
            # 3턴(사용자+AI = 6개 메시지)을 초과하는 경우, 제일 최신의 6개만 남기고 나머지는 RemoveMessage 처리
            if len(old_messages) > 6:
                msgs_to_remove = old_messages[:-6]
                trim_instructions = [RemoveMessage(id=m.id) for m in msgs_to_remove if m.id]
                
                # State에 삭제 지시사항만 있는 빈 실행을 돌려 체크포인트를 갱신합니다.
                await app_graph.aupdate_state(config, {"messages": trim_instructions})
                print(f"🧹 [Memory] 히스토리 정리 완료: {len(msgs_to_remove)}개 메시지 삭제 (최근 3턴 유지)")

        initial_state: GraphState = {
            "messages":          [HumanMessage(content=user_message)],
            "category":          "",
            "extracted_model":   "",
            "rewritten_query":   "",
            "original_question": "",   
            "context":           "",
            "generated_answer":  "",
            "is_hallucinated":   False,
            "retry_count":       0,
            "verifier_feedback": "",   
            "domain_mismatch":   False, 
            "routing_retry":     0,    
        }

        print(f"\n{'='*50}")
        print(f"[Stream] 새 요청 | thread_id: {thread_id} | user_id: {user_id}")
        print(f"[Stream] 사용자 메시지: '{user_message}'")
        if checkpoint and "messages" in checkpoint["channel_values"]:
            print(f"[Stream] 기존 대화 메시지 수: {len(checkpoint['channel_values']['messages'])}")
        print(f"{'='*50}")

        import time
        import json
        from app.core.config import get_device
        DEVICE = get_device()
        start_total = time.perf_counter()
        node_timers = {}
        streamed_nodes = set() # Track nodes that have already started streaming to avoid duplicate full text 
        yielded_text_buffer = ""

        yield json.dumps({'type': 'status', 'content': '✅ Connection established. Initializing engine...'})

        # ── astream_events: LangGraph v2 이벤트 스트리밍 ──
        # In dict inputs, LangGraph merges states. Since `messages` has `add_messages`
        # annotator, passing a new list will append to the existing threads.
        async for event in app_graph.astream_events(
            initial_state,
            config=config,
            version="v2",
        ):
            event_kind = event.get("event", "")
            node_name = event.get("metadata", {}).get("langgraph_node", "")
            if not node_name:
                continue

            if event_kind == "on_node_start":
                node_timers[node_name] = time.perf_counter()
                yield json.dumps({
                    'type': 'status', 
                    'content': f'{node_name.upper()} processing...',
                    'node': node_name
                })
            
            # ── LLM 텍스트 청크 이벤트만 필터링 ──
            if event_kind == "on_chat_model_stream":
                if node_name in ("robotics", "welding", "electrical", "general", "fallback", "social"):
                    streamed_nodes.add(node_name)
                    chunk_content = event.get("data", {}).get("chunk", {})
                    if hasattr(chunk_content, "content") and chunk_content.content:
                        text = chunk_content.content
                        
                        # LangChain V2 모델이 마지막에 전체 문장을 한 번 더 출력하는 현상 스킵
                        if len(text) > 15 and text in yielded_text_buffer:
                            continue
                            
                        yield json.dumps({'type': 'answer', 'content': text})
                        yielded_text_buffer += text

            # ── 비 스트리밍 노드(social, fallback) 완료 시 텍스트 전송 ──
            if event_kind == "on_chain_stream":
                chunk = event.get("data", {}).get("chunk", {})
                if isinstance(chunk, dict) and "generated_answer" in chunk and node_name in ("social", "fallback"):
                    if node_name not in streamed_nodes:
                        yield json.dumps({'type': 'answer', 'content': chunk['generated_answer']})
                    
            # ── 노드 완료 이벤트 ──
            elif event_kind == "on_node_end":
                elapsed = time.perf_counter() - node_timers.get(node_name, time.perf_counter())
                yield json.dumps({
                    'type': 'metadata', 
                    'content': f'{node_name.upper()} completed',
                    'node': node_name,
                    'elapsed': round(elapsed, 2)
                })
                
                if node_name == "verifier":
                    output = event.get("data", {}).get("output", {})
                    if output and output.get("is_hallucinated"):
                        yield json.dumps({'type': 'warning', 'content': 'Hallucination suspect - rewriting...'})

        total_elapsed = time.perf_counter() - start_total
        
        # 최종 상태 확인 및 메타데이터 갱신
        final_state = await app_graph.aget_state(config)
        assistant_msg = final_state.values.get("generated_answer", "")
        
        execution_metadata = {
            "total_elapsed": round(total_elapsed, 2),
            "device": DEVICE,
            "node_timings": {k: round(time.perf_counter() - v, 2) for k, v in node_timers.items()}
        }
        await app_graph.aupdate_state(config, {"metadata": execution_metadata})
        
        yield json.dumps({'type': 'status', 'content': f'✅ Finished in {total_elapsed:.2f}s'})

        # [V4.1] chat_logs RDS 로깅 연동
        from app.core.database import log_chat_interaction
        try:
            # 비동기 환경 내에서 DB 저장을 위해 to_thread 또는 직접 호출
            await asyncio.to_thread(
                log_chat_interaction, 
                user_id=user_id, 
                thread_id=thread_id, 
                query=user_message, 
                response=assistant_msg, 
                latency=total_elapsed
            )
            print("💾 [History] Chat interaction saved to chat_logs successfully.")
        except Exception as e:
            print(f"⚠️ [History] DB 저장 실패: {e}")

async def run_graph_sync(
    user_message: str,
    thread_id: str,
    user_id: str = "Unknown",
) -> str:
    """
    스트리밍 없이 그래프를 실행하고 최종 generated_answer를 반환합니다.
    테스트 또는 동기 API 엔드포인트에서 사용하세요.

    Args:
        user_message (str): 사용자 질문
        thread_id (str):    세션 ID

    Returns:
        str: 최종 생성된 답변 (fallback 포함)
    """
    config = {"configurable": {"thread_id": thread_id}}
    initial_state: GraphState = {
        "messages":          [HumanMessage(content=user_message)],
        "category":          "",
        "extracted_model":   "",
        "rewritten_query":   "",
        "original_question": "",
        "context":           "",
        "generated_answer":  "",
        "is_hallucinated":   False,
        "retry_count":       0,
        "verifier_feedback": "",
        "domain_mismatch":   False,
        "routing_retry":     0,
    }

    try:
        from app.core.history import get_async_postgres_saver
        async with get_async_postgres_saver() as saver:
            app_graph = compile_workflow(saver)
            final_state = await app_graph.ainvoke(initial_state, config=config)
            return final_state.get("generated_answer", "답변을 생성하지 못했습니다.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        return "죄송합니다. 서버 처리 중 오류가 발생했습니다."
