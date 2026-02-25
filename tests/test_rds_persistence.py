import asyncio
import os
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from langchain_core.messages import HumanMessage
from app.agents.graph import compile_workflow
from app.core.history import get_async_postgres_saver
from app.vectorstore.pgvector_store import PGVectorStoreManager

async def test_persistence():
    print("🚀 RDS 비동기 영속성 테스트 시작...")
    
    thread_id = "test_persistence_session_async_001"
    config = {"configurable": {"thread_id": thread_id}}
    
    # 1. 첫 번째 질문: 이름 기억시키기
    print("\n[Step 1] 첫 번째 질문 수행 중...")
    with PGVectorStoreManager() as _:
        async with get_async_postgres_saver() as saver:
            graph = compile_workflow(saver)
            inputs = {"messages": [HumanMessage(content="내 이름은 고길동이야. 기억해줘.")], "retry_count": 0}
            
            async for chunk in graph.astream(inputs, config=config, stream_mode="values"):
                if "generated_answer" in chunk:
                    print(f"🤖 답변 1: {chunk['generated_answer']}")

    print("\n--- 세션 종료 및 재접속 시뮬레이션 ---")
    
    # 2. 두 번째 질문: 이전 내용 기억하는지 확인
    print("\n[Step 2] 두 번째 질문 수행 중 (이전 맥락 확인)...")
    with PGVectorStoreManager() as _:
        async with get_async_postgres_saver() as saver:
            graph = compile_workflow(saver)
            inputs = {"messages": [HumanMessage(content="내 이름이 뭐라고 했지?")], "retry_count": 0}
            
            final_answer = ""
            async for chunk in graph.astream(inputs, config=config, stream_mode="values"):
                if "generated_answer" in chunk:
                    final_answer = chunk['generated_answer']
            
            print(f"🤖 답변 2: {final_answer}")
            
            if "길동" in final_answer:
                print("\n✅ 성공: RDS에서 이전 대화 맥락을 정상적으로 복구했습니다.")
            else:
                print("\n❌ 실패: 대화 맥락을 찾지 못했습니다.")

if __name__ == "__main__":
    asyncio.run(test_persistence())
