import os
import contextlib
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool

def _get_connection_string():
    """
    환경 변수와 SSH 터널링 상태를 기반으로 DB 연결 문자열을 생성합니다.
    """
    pg_user = os.getenv("PGUSER")
    pg_password = os.getenv("PGPASSWORD")
    pg_db = os.getenv("PGDATABASE")
    
    # SSH 터널링 활성화 여부 확인
    ssh_enabled = os.getenv("SSH_TUNNEL_ENABLED", "false").lower() == "true"
    ssh_local_port = os.getenv("SSH_LOCAL_BIND_PORT", "15432")
    
    if ssh_enabled:
        target_host = "127.0.0.1"
        target_port = ssh_local_port
    else:
        target_host = os.getenv("PGHOST")
        target_port = os.getenv("PGPORT", "5432")
        
    return f"postgresql://{pg_user}:{pg_password}@{target_host}:{target_port}/{pg_db}?sslmode=require"

def get_memory_saver():
    """로컬 테스트용 인메모리 세이버 (기존 호환성 유지)"""
    return MemorySaver()

@contextlib.asynccontextmanager
async def get_async_postgres_saver():
    """
    AWS RDS(Postgres)를 기반으로 하는 비동기 영속성 체크포인터를 반환합니다.
    """
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from psycopg_pool import AsyncConnectionPool
    import psycopg

    conn_info = _get_connection_string()
    
    # 1. 초기 테이블 셋업 (트랜잭션 블록 외부에서 실행)
    try:
        async with await psycopg.AsyncConnection.connect(conn_info, autocommit=True) as conn:
            saver = AsyncPostgresSaver(conn)
            await saver.setup()
    except Exception as e:
        print(f"⚠️ AsyncPostgresSaver setup 경고: {e}")

    # 2. AsyncConnectionPool을 사용하여 안정적인 연결 관리
    # min_size=1로 최소 연결 유지, timeout 상향 조정으로 SSH 터널링 지연 대응
    async with AsyncConnectionPool(
        conn_info, 
        max_size=10, 
        min_size=1, 
        timeout=30.0,
        kwargs={"connect_timeout": 10}
    ) as pool:
        async with pool.connection() as conn:
            saver = AsyncPostgresSaver(conn)
            yield saver
