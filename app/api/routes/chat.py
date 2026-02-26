from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse, JSONResponse
from app.agents.stream import stream_chat_response, run_graph_sync

router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
)

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
