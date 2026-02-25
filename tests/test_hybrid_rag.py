import sys
import os
from dotenv import load_dotenv

# 프로젝트 루트를 sys.path에 추가하여 app 모듈을 불러올 수 있게 설정
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from app.rag.retriever import get_hybrid_retriever
from app.vectorstore.pgvector_store import PGVectorStoreManager

load_dotenv()

def test_hybrid_search():
    """
    하이브리드 리트리버의 성능을 "E012", "배터리" 키워드로 테스트합니다.
    """
    print("🧪 [Test] 하이브리드 리트리버 (Vector + BM25) 성능 테스트 시작...")
    
    # 1. 하이브리드 리트리버 초기화 (BM25 가중치 0.6)
    try:
        # 하이브리드 리트리버는 벡터 데이터베이스 연결과 BM25 캐시 로드를 위해 
        # SSH 터널이 활성화된 환경에서 실행되어야 합니다 (캐시가 없을 경우 RDS 접근 필요).
        with PGVectorStoreManager(collection_name="welding_robotics_manuals") as _:
            hybrid_retriever = get_hybrid_retriever(
                collection_name="welding_robotics_manuals",
                vector_weight=0.4,
                bm25_weight=0.6
            )
            
            queries = [
                "E012 에러 조치 방법",
                "배터리 교체 매뉴얼",
                "HH012 고장 진단"
            ]
            
            for query in queries:
                print(f"\n" + "="*50)
                print(f"🔍 질문: '{query}'")
                print("="*50)
                
                # 검색 수행 (EnsembleRetriever는 invoke로 호출)
                # Note: EnsembleRetriever.invoke()는 점수(Score)를 직접 반환하지 않는 경우가 많으므로
                # 리트리버 구조를 고려하여 결과를 출력합니다.
                results = hybrid_retriever.invoke(query)
                
                if results:
                    for i, doc in enumerate(results, 1):
                        source = doc.metadata.get("source_file", "N/A")
                        h1 = doc.metadata.get("Header 1", "N/A")
                        h2 = doc.metadata.get("Header 2", "N/A")
                        
                        print(f"[{i}] [{source}] {h1} > {h2}")
                        print(f"    본문 요약: {doc.page_content[:150].replace('\n', ' ')}...")
                        print("-" * 30)
                else:
                    print("❌ 검색 결과가 없습니다.")
                    
        print("\n✅ 하이브리드 리트리버 테스트가 완료되었습니다.")

    except Exception as e:
        print(f"❌ 테스트 중 에러 발생: {e}")

if __name__ == "__main__":
    test_hybrid_search()
