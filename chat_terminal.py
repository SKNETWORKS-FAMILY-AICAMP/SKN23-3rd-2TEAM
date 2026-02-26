import asyncio
import sys
import os
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가하여 app 모듈 인식
ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))

from langchain_core.messages import HumanMessage
from app.agents.graph import compile_workflow
from app.core.history import get_async_postgres_saver
from app.vectorstore.pgvector_store import PGVectorStoreManager

async def terminal_chat():
    print("\n" + "="*50)
    print("🚀 산업용 기술지원 챗봇 [터미널 디버그 모드] 시작")
    print("   (AWS RDS 영속성 및 SSH 터널링 활성화)")
    print("="*50)
    
    thread_id = "terminal_debug_session_001"
    
    try:
        # 1. SSH 터널 및 DB 세션 유지 (Vector Store용)
        with PGVectorStoreManager() as _:
            print(f"✅ AWS RDS 연결 성공 (Session: {thread_id})")
            
            # 2. AsyncPostgresSaver 연동
            async with get_async_postgres_saver() as saver:
                # RDS 체크포인터가 지정된 그래프 컴파일
                app = compile_workflow(saver)
                
                while True:
                    print("\n" + "-"*30)
                    sys.stdout.write("👤 사용자: ")
                    sys.stdout.flush()
                    # [FIX] Encoding robustness for various terminal environments
                    raw_input = sys.stdin.buffer.readline()
                    if not raw_input: # EOF
                        break
                    user_input = raw_input.decode('utf-8', errors='replace').strip()
                    if user_input.lower() in ["exit", "quit", "종료", "q"]: 
                        break
                    
                    if not user_input:
                        continue

                    # 3. LangGraph 실행 (astream_events v2 사용)
                    config = {"configurable": {"thread_id": thread_id}}
                    
                    print("🤖 답변: ", end="", flush=True)
                    
                    # GraphState 초기값 설정
                    inputs = {"messages": [HumanMessage(content=user_input)], "retry_count": 0}
                    
                    async for event in app.astream_events(
                        inputs, 
                        config, 
                        version="v2"
                    ):
                        # 노드 전이 로그 (디버깅 시 활성화 가능)
                        # if event["event"] == "on_chain_start" and event["name"] in ["rewriter", "supervisor", "robotics", "welding", "electrical", "verifier"]:
                        #     print(f"\n🔄 [Node] {event['name']} 실행 중...")
                        
                        # 최종 답변 스트리밍
                        if event["event"] == "on_chat_model_stream":
                            content = event["data"]["chunk"].content
                            if content:
                                print(content, end="", flush=True)
                    print() # 한 줄 띄우기

    except KeyboardInterrupt:
        print("\n\n👋 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
    finally:
        print("\n" + "="*50)
        print("🛑 터미널 챗봇 모드를 종료합니다.")
        print("="*50)

if __name__ == "__main__":
    # Windows 환경에서 ProactorEventLoop 관련 경고 방지
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(terminal_chat())
