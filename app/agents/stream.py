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
                
                # 프론트엔드 진행바(Status)를 위한 사용자 친화적 메시지 매핑
                status_msg = f"{node_name.upper()} processing..."
                if node_name in ("welding", "robotics", "electrical", "general"):
                    status_msg = "🔍 관련된 전문 매뉴얼을 검색하고 분석하는 중입니다..."
                elif node_name == "verifier":
                    status_msg = "🛡️ 답변에 잘못된 정보(환각)가 있는지 검증하고 있습니다..."
                elif node_name == "feedback_rewriter":
                    status_msg = "🔄 피드백을 반영하여 더 정확하고 안전한 답변으로 재작성 중입니다..."
                elif node_name == "router":
                    status_msg = "🚦 질문의 의도를 분석하여 적절한 에이전트를 할당 중입니다..."
                elif node_name == "social":
                    status_msg = "💬 일상 대화에 답변하는 중입니다..."
                
                yield json.dumps({
                    'type': 'status', 
                    'content': status_msg,
                    'node': node_name
                })
            
            # ── [수정] 중간 노드의 LLM 텍스트 스트리밍 제거 ──
            # (환각 검증/재작성 루프에서 텍스트가 덮어씌워지는 문제 해결을 위해, 
            #  최종 그래프 완료 후 한 번에(또는 그때부터) 답변을 스트리밍합니다.)
            
            # ── 비 스트리밍 노드(social, fallback) 완료 시 텍스트 전송 (이제 최종 전송으로 통합) ──
                    
            # ── 노드 완료 이벤트 ──
            elif event_kind == "on_node_end":
                elapsed = time.perf_counter() - node_timers.get(node_name, time.perf_counter())
                if node_name == "verifier":
                    output = event.get("data", {}).get("output", {})
                    if output and output.get("is_hallucinated"):
                        yield json.dumps({'type': 'status', 'content': '⚠ 환각(거짓 정보) 요소가 발견되어 안전하게 검토 중입니다...'})

        total_elapsed = time.perf_counter() - start_total
        
        # ── [수정] 최종 상태 확인 및 답변 스트리밍 ──
        final_state = await app_graph.aget_state(config)
        assistant_msg = final_state.values.get("generated_answer", "")
        
        # 그래프 실행이 끝나면 상태바를 완료 처리하기 위한 상태 전송
        yield json.dumps({'type': 'status_complete', 'content': f'✅ 답변 준비 완료 ({total_elapsed:.2f}s)'})
        
        # 최종 확정된 답변을 청크 단위로 나누어 스트리밍 (부드러운 UI 렌더링 효과)
        if assistant_msg:
            chunk_size = 15  # 한 번에 보낼 문자 수
            for i in range(0, len(assistant_msg), chunk_size):
                chunk_piece = assistant_msg[i:i+chunk_size]
                yield json.dumps({'type': 'answer', 'content': chunk_piece})
                await asyncio.sleep(0.01) # 부드러운 스트리밍 타이밍 간격
        
        execution_metadata = {
            "total_elapsed": round(total_elapsed, 2),
            "device": DEVICE,
            "node_timings": {k: round(time.perf_counter() - v, 2) for k, v in node_timers.items()}
        }
        await app_graph.aupdate_state(config, {"metadata": execution_metadata})

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
