"""
backend/sse_timing_guide.py
────────────────────────────────────────────────────────────────
백엔드에서 프론트엔드로 전송해야 하는 SSE 이벤트 포맷 가이드.
main.py 의 /chat 엔드포인트에 통합하세요.

프론트엔드(main_ui.py)는 아래 4가지 type 을 파싱합니다:
    node_start  → 노드 시작 (타이머 시작)
    node_end    → 노드 종료 (소요 시간 계산)
    answer      → 스트리밍 답변 토큰
    status      → 일반 상태 메시지 (선택)
    warning     → 경고
    error       → 오류
"""

import asyncio
import json
import time
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()


def sse(payload: dict) -> str:
    """SSE 프레임 직렬화 헬퍼."""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def run_langgraph_stream(message: str, thread_id: str):
    """
    LangGraph astream_events 를 순회하며 SSE 이벤트를 yield 합니다.

    on_chain_start / on_chain_end 이벤트에서 node 이름을 추출하고
    perf_counter 로 소요 시간을 측정한 뒤 node_start / node_end 를 전송합니다.
    """
    # 노드 → 표준 이름 매핑 (LangGraph 내부 체인 이름 → 프론트엔드 key)
    NODE_MAP = {
        "rewriter_chain":   "rewriter",
        "retriever_chain":  "retriever",
        "reranker_chain":   "reranker",
        "specialist_chain": "specialist",
        "verifier_chain":   "verifier",
    }

    node_start_times: dict[str, float] = {}

    # ── 예시: your_graph 는 실제 컴파일된 LangGraph 인스턴스
    # async for event in your_graph.astream_events(
    #     {"input": message},
    #     config={"configurable": {"thread_id": thread_id}},
    #     version="v2",
    # ):

    # ────────── 아래는 실제 이벤트 처리 로직 예시 ──────────
    async for event in _PLACEHOLDER_stream(message):
        event_name = event.get("event", "")
        run_name   = event.get("name", "")

        node_key = NODE_MAP.get(run_name)

        # ── 노드 시작
        if event_name == "on_chain_start" and node_key:
            node_start_times[node_key] = time.perf_counter()
            yield sse({"type": "node_start", "node": node_key})

        # ── 노드 종료
        elif event_name == "on_chain_end" and node_key:
            elapsed = time.perf_counter() - node_start_times.get(node_key, time.perf_counter())
            yield sse({"type": "node_end", "node": node_key, "elapsed": round(elapsed, 3)})

        # ── LLM 스트리밍 청크 → answer 이벤트
        elif event_name == "on_chat_model_stream":
            chunk_text = event.get("data", {}).get("chunk", {}).get("content", "")
            if chunk_text:
                # 누적 answer 를 보내거나 delta 를 보낼 수 있음
                # 프론트엔드는 content 를 그대로 덮어쓰므로 누적 전송 권장
                yield sse({"type": "answer", "content": chunk_text})

        # ── 기타 상태 메시지
        elif event_name == "on_custom_event":
            yield sse({"type": "status", "content": event.get("data", {}).get("message", "")})


async def _PLACEHOLDER_stream(message: str):
    """
    실제 LangGraph 스트림 대신 사용하는 플레이스홀더.
    실제 구현 시 이 함수를 제거하고 your_graph.astream_events(...)를 사용하세요.
    """
    nodes = [
        "rewriter_chain", "retriever_chain",
        "reranker_chain", "specialist_chain", "verifier_chain",
    ]
    for n in nodes:
        yield {"event": "on_chain_start", "name": n}
        await asyncio.sleep(0.4)   # 시뮬레이션
        yield {"event": "on_chain_end",   "name": n}
    for word in ["안녕하세요! ", "질문을 ", "처리했습니다."]:
        yield {"event": "on_chat_model_stream", "data": {"chunk": {"content": word}}}
        await asyncio.sleep(0.05)


@app.post("/chat")
async def chat_endpoint(body: dict):
    message   = body.get("message", "")
    thread_id = body.get("thread_id", "default")

    return StreamingResponse(
        run_langgraph_stream(message, thread_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # Nginx 프록시 버퍼링 비활성화
        },
    )