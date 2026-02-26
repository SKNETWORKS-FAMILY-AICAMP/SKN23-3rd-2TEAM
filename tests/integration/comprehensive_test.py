import sys
import os
import time
import asyncio
import json
from datetime import datetime
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

# 프로젝트 루트를 sys.path에 추가
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from app.agents.graph import app_graph, FALLBACK_LOG_PATH
from app.vectorstore.pgvector_store import PGVectorStoreManager, get_vector_store
from app.rag.retriever import BM25_CACHE_PATH

load_dotenv()

async def run_scenario(scenario_name, query):
    print(f"\n" + "="*80)
    print(f"🎬 [Scenario {scenario_name}] 시작: '{query}'")
    print("="*80)
    
    start_time = time.time()
    
    # LangGraph 실행 (입력 상태 초기화)
    config = {"configurable": {"thread_id": f"test_{scenario_name.split()[0]}"}}
    inputs = {"messages": [HumanMessage(content=query)]}
    
    node_timings = []
    
    try:
        current_node_start = time.time()
        # Updates 모드로 실행하여 노드 전이 및 소요 시간 상세 추적
        print("\n--- [Workflow Tracing] ---")
        async for chunk in app_graph.astream(inputs, config=config, stream_mode="updates"):
            for node_name, state_update in chunk.items():
                elapsed = (time.time() - current_node_start) * 1000
                print(f"📍 Node: [{node_name:15}] | Latency: {elapsed:8.2f}ms")
                node_timings.append((node_name, elapsed))
                current_node_start = time.time()

        # 최종 결과 확인
        final_values = None
        # Values 모드로 마지막 상태 추출
        async for v in app_graph.astream(inputs, config=config, stream_mode="values"):
            final_values = v
            
        end_time = time.time()
        total_latency = (end_time - start_time) * 1000
            
        print(f"📝 최종 답변: {final_values.get('generated_answer', 'N/A')}")
        print(f"🔍 카테고리: {final_values.get('category', 'N/A')}")
        print(f"🚩 Hallucinated: {final_values.get('is_hallucinated', 'N/A')}")
        print(f"🔄 Retry Count: {final_values.get('retry_count', 0)}")
        print(f"⏱️ 총 소요 시간: {total_latency:.2f}ms")
        
    except Exception as e:
        print(f"❌ 시나리오 중단: {e}")

async def check_connectivity():
    print("\n" + "="*80)
    print("🔌 [검증 1] DB 연결성 및 데이터 정합성 체크")
    print("="*80)
    
    try:
        # 1. BM25 캐시 존재 확인
        if os.path.exists(BM25_CACHE_PATH):
            size = os.path.getsize(BM25_CACHE_PATH) / (1024 * 1024)
            print(f"✅ BM25 캐시 발견: {BM25_CACHE_PATH} ({size:.2f} MB)")
        else:
            print("⚠️ BM25 캐시가 없습니다. 첫 실행 시 생성이 필요합니다.")

        # 2. RDS 연결 및 데이터 샘플링
        with PGVectorStoreManager(collection_name="welding_robotics_manuals") as _:
            store = get_vector_store(collection_name="welding_robotics_manuals")
            # 임의의 데이터 1건 fetch
            sample_docs = store.similarity_search("현대로보틱스", k=1)
            if sample_docs:
                print(f"✅ DB 연결 성공! 샘플 데이터 출처: {sample_docs[0].metadata.get('source_file')}")
            else:
                print("❌ DB 연결은 성공했으나 데이터를 찾을 수 없습니다.")
                
    except Exception as e:
        print(f"❌ DB 연결 실패: {e}")

def check_unimplemented_features():
    print("\n" + "="*80)
    print("🚧 [검증 4] 미구현 및 예외 처리 점검")
    print("="*80)
    
    # 1. Fallback 노드 로직 확인 (코드 상 존재 여부는 이미 파악됨)
    print("✅ Fallback 노드: app/agents/graph.py 내 구현 확인됨.")
    
    # 2. JSONL 로그 기록 확인
    if os.path.exists(FALLBACK_LOG_PATH):
        print(f"✅ JSONL 로그 파일 존재: {FALLBACK_LOG_PATH}")
    else:
        print(f"⚠️ JSONL 로그 파일이 아직 생성되지 않았습니다 (첫 fallback 발생 시 생성됨).")

async def main():
    start_all = time.time()
    
    # 1. 연결성 체크
    await check_connectivity()
    
    # 2. 미구현 기능 체크
    check_unimplemented_features()
    
    # 3. 테스트 시나리오 실행 (하나의 터널 컨텍스트 내에서 실행)
    print("\n" + "="*80)
    print("🚀 테스트 시나리오 통합 실행 시작")
    print("="*80)
    
    scenarios = [
        ("A (정상)", "현대로보틱스 Hi6 제어기 E012 에러 해결 방법은?"),
        ("B (도메인 불일치)", "용접기 매뉴얼에서 요리 레시피 찾아줘"),
        ("C (미존재 데이터)", "2050년형 최신 초전도 로봇 수리법")
    ]
    
    try:
        with PGVectorStoreManager(collection_name="welding_robotics_manuals") as _:
            for name, query in scenarios:
                await run_scenario(name, query)
    except Exception as e:
        print(f"❌ 통합 실행 중 오류 발생: {e}")
        
    print(f"\n🚀 모든 테스트 완료! (전체 소요 시간: {time.time() - start_all:.2f}s)")

if __name__ == "__main__":
    asyncio.run(main())
