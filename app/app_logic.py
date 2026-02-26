import asyncio
import json
from typing import AsyncGenerator
from langchain_core.messages import HumanMessage
from app.agents.graph import compile_workflow
from app.core.history import get_async_postgres_saver
from app.vectorstore.pgvector_store import PGVectorStoreManager

async def run_langgraph_stream(query: str, thread_id: str) -> AsyncGenerator[str, None]:
    """
    LangGraph를 실행하고 발생하는 이벤트를 실시간으로 yield 합니다.
    """
    # 1. SSH 터널 및 RDS 체크포인터 연결
    with PGVectorStoreManager() as _:
        async with get_async_postgres_saver() as saver:
            app = compile_workflow(saver)
            
            config = {"configurable": {"thread_id": thread_id}}
            inputs = {"messages": [HumanMessage(content=query)]}
            
            # v2 astream_events 사용
            async for event in app.astream_events(inputs, config, version="v2"):
                kind = event["event"]
                
                # 1. 노드 전이 상태 (Processing...)
                if kind == "on_node_start":
                    node_name = event["metadata"].get("langgraph_node", "")
                    if node_name:
                        yield f"STATUS:{node_name.upper()} processing..."
                
                # 2. 답변 스트리밍 (Token by Token)
                elif kind == "on_chat_model_stream":
                    content = event["data"]["chunk"].content
                    if content:
                        yield content
                
                # 3. 노드 종료 및 최종 결과 (선택적)
                elif kind == "on_node_end":
                    # 필요시 추가 정보 전송
                    pass

async def get_streaming_response(query: str, thread_id: str):
    """
    Streamlit에서 사용할 수 있도록 비동기 제너레이터를 감쌉니다.
    """
    async for chunk in run_langgraph_stream(query, thread_id):
        yield chunk
