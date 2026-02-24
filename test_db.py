import os
from sqlalchemy import create_engine, text
from app.infrastructure.aws.rds_pgvector_client import get_pgvector_url

def test_connection():
    print("🚀 DB 연결 테스트를 시작합니다 (Bastion SSH 터널링 경유)...")
    try:
        # 방금 만든 연결 문자열 생성기(컨텍스트 매니저) 사용
        with get_pgvector_url() as db_url:
            # 보안을 위해 비밀번호 부분은 마스킹 처리해서 출력
            safe_url = db_url.replace('Enc0re!2026', '********')
            print(f"🔗 터널링 성공! 생성된 DB URL: {safe_url}")
            
            # SQLAlchemy로 실제 쿼리 날려보기
            engine = create_engine(db_url)
            with engine.connect() as conn:
                result = conn.execute(text("SELECT 1;")).scalar()
                
                if result == 1:
                    print("✅ [성공] Bastion Host를 통과하여 RDS PostgreSQL에 완벽하게 연결되었습니다!")
                    print("✅ [성공] 응답 쿼리 결과: 1")
                    
    except Exception as e:
        print(f"❌ [실패] 연결 중 오류가 발생했습니다.\n에러 내용: {e}")

if __name__ == "__main__":
    test_connection()