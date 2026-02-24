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
from app.agents.graph import app_graph
from app.schemas.state import GraphState

async def stream_chat_response(
    user_message: str,
    thread_id: str,
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

    initial_state: GraphState = {
        "messages":          [HumanMessage(content=user_message)],
        "category":          "",
        "extracted_model":   "",
        "rewritten_query":   "",
        "original_question": "",   # [FIX] feedback 루프 앵커 필드
        "context":           "",
        "generated_answer":  "",
        "is_hallucinated":   False,
        "retry_count":       0,
        "verifier_feedback": "",   # [FIX] Verifier 피드백 필드
        "domain_mismatch":   False, # [FIX] 도메인 불일치 플래그
        "routing_retry":     0,    # [FIX] 재분류 회수 카운터
    }

    print(f"\n{'='*50}")
    print(f"[Stream] 새 요청 | thread_id: {thread_id}")
    print(f"[Stream] 사용자 메시지: '{user_message}'")
    print(f"{'='*50}")

    # ── astream_events: LangGraph v2 이벤트 스트리밍 ──
    # 각 노드의 실행 이벤트 및 LLM 청크를 실시간으로 수신합니다.
    async for event in app_graph.astream_events(
        initial_state,
        config=config,
        version="v2",    # LangGraph astream_events API 버전
    ):
        event_kind = event.get("event", "")
        node_name = event.get("metadata", {}).get("langgraph_node", "")

        # ── LLM 텍스트 청크 이벤트만 필터링 ──
        # "on_chat_model_stream": LLM이 청크를 생성할 때마다 발생
        if event_kind == "on_chat_model_stream":
            # specialist 노드(robotics/welding/electrical) 또는 general 노드에서 온 청크만 처리
            # rewriter, supervisor, verifier의 출력은 최종 답변이 아니므로 제외
            if node_name in ("robotics", "welding", "electrical", "general", "fallback"):
                chunk_content = event.get("data", {}).get("chunk", {})
                if hasattr(chunk_content, "content") and chunk_content.content:
                    yield chunk_content.content
                    # 청크가 비어있지 않을 때만 Yield
                    # FastAPI StreamingResponse가 이 값을 즉시 클라이언트에 전송

        # ── 노드 완료 이벤트 ── (로깅용)
        elif event_kind == "on_chain_end":
            output = event.get("data", {}).get("output", {})
            if node_name == "fallback":
                # fallback 노드는 스트리밍이 아니므로 완료 후 전체 메시지를 한 번에 Yield
                fallback_msg = output.get("generated_answer", "")
                if fallback_msg:
                    yield fallback_msg

async def run_graph_sync(
    user_message: str,
    thread_id: str,
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
        "original_question": "",   # [FIX] feedback 루프 앵커 필드
        "context":           "",
        "generated_answer":  "",
        "is_hallucinated":   False,
        "retry_count":       0,
        "verifier_feedback": "",   # [FIX] Verifier 피드백 필드
        "domain_mismatch":   False, # [FIX] 도메인 불일치 플래그
        "routing_retry":     0,    # [FIX] 재분류 회수 카운터
    }

    # ainvoke: 그래프 전체를 비동기로 실행 후 최종 state 반환
    final_state = await app_graph.ainvoke(initial_state, config=config)
    return final_state.get("generated_answer", "답변을 생성하지 못했습니다.")
