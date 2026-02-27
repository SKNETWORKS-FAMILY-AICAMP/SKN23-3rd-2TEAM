from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse, JSONResponse
from app.agents.stream import stream_chat_response, run_graph_sync

router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
)

from pydantic import BaseModel

class ChatRequest(BaseModel):
    message: str
    thread_id: str
    user_id: str = "Unknown"

@router.get(
    "/stream",
    summary="실시간 스트리밍 답변 (SSE)",
    description="LLM이 생성하는 텍스트 청크를 토큰 단위로 실시간 전송합니다. (text/event-stream)",
)
async def chat_stream_endpoint(
    q: str = Query(..., description="사용자 질문 (예: E012 에러)"),
    thread_id: str = Query("default", description="세션 ID (멀티턴 대화 이력 분리)"),
):
    """Server-Sent Events (SSE) 방식으로 스트리밍 답변을 제공합니다."""
    async def event_generator():
        async for chunk in stream_chat_response(q, thread_id):
            yield f"data: {chunk}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )

@router.get(
    "",
    summary="단건 JSON 답변",
    description="스트리밍 없이 최종 답변만 JSON으로 반환합니다. (테스트 및 배치 호출용)",
)
async def chat_endpoint(
    q: str = Query(..., description="사용자 질문"),
    thread_id: str = Query("default", description="세션 ID"),
):
    """
    전체 그래프가 완료된 뒤 최종 generated_answer를 JSON으로 반환합니다.
    스트리밍이 필요 없는 배치 처리나 API 테스트에 활용하세요.
    """
    try:
        answer = await run_graph_sync(q, thread_id)
        return JSONResponse(content={
            "thread_id": thread_id,
            "question": q,
            "answer": answer,
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "message": "그래프 실행 중 오류가 발생했습니다."}
        )

@router.post(
    "",
    summary="JSON Body 기반 실시간 스트리밍 답변 (SSE)",
    description="Streamlit UI 및 클라이언트에서 사용하는 통합 스트리밍 엔드포인트입니다.",
)
async def chat_post_stream(request: ChatRequest):
    """
    POST 요청으로 메시지를 전달받아 SSE(Server-Sent Events) 형식으로 답변을 반환합니다.
    """
    async def event_generator():
        try:
            async for chunk in stream_chat_response(request.message, request.thread_id, request.user_id):
                yield f"data: {chunk}\n\n"
        except Exception as e:
            import json
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'content': f'Engine Error: {str(e)}'})}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )

@router.get("/history/{thread_id}")
async def get_history(thread_id: str):
    try:
        from app.core.history import get_async_postgres_saver
        async with get_async_postgres_saver() as saver:
            config = {"configurable": {"thread_id": thread_id}}
            checkpoint = await saver.aget(config)
            if checkpoint and "messages" in checkpoint["channel_values"]:
                msgs = checkpoint["channel_values"]["messages"]
                return [{"role": "user" if m.type=="human" else "assistant", "content": m.content} for m in msgs]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return []

