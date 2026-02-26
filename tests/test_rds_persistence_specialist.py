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

async def test_persistence_specialist():
    print("🚀 RDS 비동기 영속성 테스트 (Specialist 모드) 시작...")
    
    thread_id = "test_robot_session_002"
    config = {"configurable": {"thread_id": thread_id}}
    
    # 1. 첫 번째 질문: 로봇 관련 에러
    print("\n[Step 1] 첫 번째 질문 수행 중 (Hi6 E012)...")
    with PGVectorStoreManager() as _:
        async with get_async_postgres_saver() as saver:
            graph = compile_workflow(saver)
            inputs = {"messages": [HumanMessage(content="현대로보틱스 Hi6 제어기 E012 에러가 뭐야?")], "retry_count": 0}
            
            async for chunk in graph.astream(inputs, config=config, stream_mode="values"):
                pass
            print("🤖 답변 1 완료.")

    # 2. 두 번째 질문: "이전 에러 해결 방법은?" (맥락 유지 확인)
    print("\n[Step 2] 두 번째 질문 수행 중 (해결 방법 문의)...")
    with PGVectorStoreManager() as _:
        async with get_async_postgres_saver() as saver:
            graph = compile_workflow(saver)
            inputs = {"messages": [HumanMessage(content="해당 에러의 해결 방법은 뭐야?")], "retry_count": 0}
            
            final_answer = ""
            async for chunk in graph.astream(inputs, config=config, stream_mode="values"):
                if "generated_answer" in chunk:
                    final_answer = chunk['generated_answer']
            
            print(f"🤖 답변 2: {final_answer[:100]}...")
            
            # 답변에 에러 해결 관련 키워드가 있는지 확인
            if "조치" in final_answer or "확인" in final_answer or "해결" in final_answer:
                print("\n✅ 성공: 이전 질문(E012 에러)의 맥락을 유지하여 해결 방법을 제시했습니다.")
            else:
                print("\n❌ 실패: 이전 질문의 맥락을 인계받지 못했을 가능성이 높습니다.")

if __name__ == "__main__":
    asyncio.run(test_persistence_specialist())
