# ============================================================
# [비즈니스 로직 서비스 레이어] app/services/chat_service.py
# ============================================================
# FastAPI 라우트와 LangGraph 그래프 사이의 비즈니스 로직을 처리합니다.
# ============================================================
from typing import AsyncGenerator
from app.agents.stream import stream_chat_response, run_graph_sync


async def handle_chat_stream(
    user_message: str,
    thread_id: str,
) -> AsyncGenerator[str, None]:
    """
    스트리밍 응답 서비스 함수.
    FastAPI 라우트에서 호출 — SSE (Server-Sent Events) 스트리밍용.
    
    Args:
        user_message: 사용자 질문
        thread_id: 세션 ID (멀티턴 이력 분리)
    
    Yields:
        str: LLM이 생성하는 텍스트 청크
    """
    async for chunk in stream_chat_response(user_message, thread_id):
        yield chunk


async def handle_chat_sync(
    user_message: str,
    thread_id: str,
) -> str:
    """
    단건 동기형 응답 서비스 함수.
    FastAPI 라우트에서 호출 — JSON 응답 반환용.
    
    Args:
        user_message: 사용자 질문
        thread_id: 세션 ID

    Returns:
        str: 최종 생성된 답변 텍스트
    """
    return await run_graph_sync(user_message, thread_id)
