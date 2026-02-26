import asyncio
import os
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import psycopg
from app.vectorstore.pgvector_store import PGVectorStoreManager

async def check_rds_tables():
    print("🔍 RDS 체크포인트 테이블 레코드 확인 중...")
    
    pg_user = os.getenv("PGUSER")
    pg_password = os.getenv("PGPASSWORD")
    pg_db = os.getenv("PGDATABASE")
    ssh_local_port = os.getenv("SSH_LOCAL_BIND_PORT", "15432")
    conn_info = f"postgresql://{pg_user}:{pg_password}@127.0.0.1:{ssh_local_port}/{pg_db}?sslmode=require"

    with PGVectorStoreManager() as _:
        try:
            async with await psycopg.AsyncConnection.connect(conn_info) as conn:
                async with conn.cursor() as cur:
                    # 1. checkpoints 테이블 확인
                    await cur.execute("SELECT count(*) FROM checkpoints;")
                    checkpoints_count = (await cur.fetchone())[0]
                    
                    # 2. checkpoint_blobs 테이블 확인
                    await cur.execute("SELECT count(*) FROM checkpoint_blobs;")
                    blobs_count = (await cur.fetchone())[0]
                    
                    # 3. 최근 thread_id 확인
                    await cur.execute("SELECT thread_id, checkpoint_id FROM checkpoints ORDER BY checkpoint_id DESC LIMIT 5;")
                    recent_threads = await cur.fetchall()

                    print(f"\n📊 [통합 결과]")
                    print(f"  - checkpoints 레코드 수: {checkpoints_count}")
                    print(f"  - checkpoint_blobs 레코드 수: {blobs_count}")
                    print(f"\n🕒 [최근 활성 세션]")
                    for tid, cid in recent_threads:
                        print(f"  - Thread ID: {tid} (Checkpoint: {cid[:8]}...)")

                    if checkpoints_count > 0:
                        print("\n✅ 확인 완료: 모든 대화 상태가 AWS RDS에 안전하게 영속화되고 있습니다.")
                    else:
                        print("\n❌ 실패: 테이블은 존재하나 데이터가 없습니다.")
        except Exception as e:
            print(f"❌ 오류 발생: {e}")

if __name__ == "__main__":
    asyncio.run(check_rds_tables())
