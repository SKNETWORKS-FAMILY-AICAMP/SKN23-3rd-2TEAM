import sys
import os

# 프로젝트 루트를 sys.path에 추가하여 app 모듈을 불러올 수 있게 설정
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from app.vectorstore.pgvector_store import PGVectorStoreManager

def verify_pgvector_search():
    """
    RDS pgvector에 데이터가 성공적으로 적재되었는지 검색을 통해 검증합니다.
    SSH 터널링은 PGVectorStoreManager가 자동으로 처리합니다.
    """
    print("🔍 [Verification] RDS pgvector 검색 검증 시작...")
    
    # 검색어 목록
    query_list = ["E012", "배터리", "용접 스패터"]
    
    try:
        with PGVectorStoreManager(collection_name="welding_robotics_manuals") as vector_store:
            for query in query_list:
                print(f"\n--- 검색어: '{query}' ---")
                
                # 유사도 검색 수행 (k=3)
                results = vector_store.similarity_search_with_score(query, k=3)
                
                if results:
                    for i, (doc, score) in enumerate(results, 1):
                        print(f"[{i}] 유사도 거리: {score:.4f}")
                        print(f"    출처: {doc.metadata.get('source_file', 'N/A')}")
                        print(f"    본문: {doc.page_content[:150].replace('\n', ' ')}...")
                        print("-" * 30)
                else:
                    print(f"❌ '{query}'에 대한 검색 결과가 없습니다.")
                    
        print("\n✅ 모든 검증 단계가 종료되었습니다.")

    except Exception as e:
        print(f"❌ 검증 중 에러 발생: {e}")

if __name__ == "__main__":
    verify_pgvector_search()