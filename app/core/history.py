import os
import contextlib
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool

def _get_conn_info():
    """환경 변수를 사용하여 PostgreSQL 연결 정보를 구성합니다."""
    host = os.getenv("PGHOST", "localhost")
    user = os.getenv("PGUSER", "postgres")
    pw = os.getenv("PGPASSWORD", "password")
    db = os.getenv("PGDATABASE", "chatbot_db")
    port = os.getenv("PGPORT", "5432")
    
    # SSH 터널링 포트 적용 (필요 시)
    ssh_enabled = os.getenv("SSH_TUNNEL_ENABLED", "false").lower() == "true"
    if ssh_enabled:
        host = "127.0.0.1"
        port = os.getenv("SSH_LOCAL_BIND_PORT", "15432")
        
    # TCP Keepalives 설정 추가 (타임아웃 방지)
    keepalives = "keepalives=1 keepalives_idle=60 keepalives_interval=10 keepalives_count=5"
    
    return f"host={host} user={user} password={pw} dbname={db} port={port} sslmode=require {keepalives}"

def get_memory_saver():
    """
    [V4.0] PostgresSaver를 활용한 전역 체크포인터 설정.
    이 함수는 동기식 ConnectionPool을 사용하여 Saver를 구성하고 초기화합니다.
    """
    conninfo = _get_conn_info()
    
    # 동기식 ConnectionPool 설정 (max_lifetime 으로 Stale Connection 방지)
    pool = ConnectionPool(conninfo, max_size=10, min_size=1, max_lifetime=300)
    
    # PostgresSaver 생성
    checkpointer = PostgresSaver(pool)
    
    # [핵심] 삭제된 체크포인트 테이블 자동 재구축 (Internal setup)
    # v4.0에서 테이블을 Drop했으므로 반드시 호출해야 합니다.
    checkpointer.setup()
    
    return checkpointer

@contextlib.asynccontextmanager
async def get_async_postgres_saver():
    """
    비동기 전용 PostgresSaver를 위한 컨텍스트 매니저 (기존 비동기 코드 호환)
    """
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from psycopg_pool import AsyncConnectionPool
    
    conninfo = _get_conn_info()
    
    # max_lifetime 설정으로 연결 재활용(Recycle) 활성화하여 Timeout 차단
    async with AsyncConnectionPool(conninfo, max_size=10, max_lifetime=300) as pool:
        async with pool.connection() as conn:
            checkpointer = AsyncPostgresSaver(conn)
            # 비동기 setup 호출
            await checkpointer.setup()
            yield checkpointer
