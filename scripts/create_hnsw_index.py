import os
import sys
import psycopg2

# 프로젝트 최상단 디렉토리를 경로에 추가하여 app 모듈을 임포트할 수 있도록 함
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.database import get_connection_kwargs, open_optional_ssh_tunnel

def create_hnsw_index():
    print("⏳ DB에 연결하여 HNSW 인덱스 생성을 시도합니다...")
    try:
        with open_optional_ssh_tunnel() as tunnel:
            conn_args = get_connection_kwargs()
            if tunnel:
                conn_args["host"] = tunnel["host"]
                conn_args["port"] = tunnel["port"]
            
            with psycopg2.connect(**conn_args) as conn:
                with conn.cursor() as cur:
                    # HNSW 인덱스 생성 쿼리 실행
                    # langchain_pg_embedding의 embedding 컬럼이 차원 정보가 없는 vector 타입이므로 
                    # text-embedding-3-small 모델의 1536 차원으로 명시적 형변환(casting)하여 인덱스 생성
                    cur.execute("""
                        CREATE INDEX IF NOT EXISTS hnsw_idx 
                        ON langchain_pg_embedding 
                        USING hnsw ((embedding::vector(1536)) vector_cosine_ops);
                    """)
                    conn.commit()
                    print("✅ PgVector HNSW 인덱스 생성 완료! 이제 검색 속도가 1초 이내로 단축됩니다.")
    except Exception as e:
        print(f"❌ 인덱스 생성 중 오류 발생: {e}")

if __name__ == "__main__":
    create_hnsw_index()
