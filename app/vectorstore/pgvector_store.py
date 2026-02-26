import os
import sys
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.pgvector import PGVector
from sshtunnel import SSHTunnelForwarder

load_dotenv()

class PGVectorStoreManager:
    """
    SSH 터널링을 자동으로 관리하고 PGVector 인스턴스를 제공하는 클래스입니다.
    """
    def __init__(self, collection_name: str = "welding_robotics_manuals"):
        self.collection_name = collection_name
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.tunnel = None
        self.vector_store = None

    def _get_env_or_raise(self, key):
        value = os.getenv(key)
        if not value:
            raise ValueError(f"환경 변수 {key}가 설정되지 않았습니다.")
        return value

    def __enter__(self):
        ssh_enabled = os.getenv("SSH_TUNNEL_ENABLED", "false").lower() == "true"
        pg_host = self._get_env_or_raise("PGHOST")
        pg_port = int(os.getenv("PGPORT", "5432"))
        pg_user = self._get_env_or_raise("PGUSER")
        pg_password = self._get_env_or_raise("PGPASSWORD")
        pg_db = self._get_env_or_raise("PGDATABASE")

        # 🌟 핵심: 터널링 포트 충돌 방지를 위해 is_port_in_use 도입
        from app.core.database import is_port_in_use

        if ssh_enabled:
            ssh_host = self._get_env_or_raise("SSH_HOST")
            ssh_port = int(os.getenv("SSH_PORT", "22"))
            ssh_user = self._get_env_or_raise("SSH_USER")
            ssh_key_path = self._get_env_or_raise("SSH_PRIVATE_KEY_PATH")
            local_bind_port = int(os.getenv("SSH_LOCAL_BIND_PORT", "15432"))

            # 포트가 이미 사용 중이라면(run_tunnel.py 구동 중) 터널을 열지 않음
            if is_port_in_use(local_bind_port):
                target_host = '127.0.0.1'
                target_port = local_bind_port
                print(f"🔗 Reusing existing SSH Tunnel on {target_host}:{target_port}")
            else:
                self.tunnel = SSHTunnelForwarder(
                    (ssh_host, ssh_port),
                    ssh_username=ssh_user,
                    ssh_pkey=ssh_key_path,
                    remote_bind_address=(pg_host, pg_port),
                    local_bind_address=('127.0.0.1', local_bind_port)
                )
                self.tunnel.start()
                target_host = '127.0.0.1'
                target_port = self.tunnel.local_bind_port
                print(f"🔗 New SSH Tunnel established on {target_host}:{target_port}")
        else:
            target_host = pg_host
            target_port = pg_port
            print(f"🌐 Direct connection to {target_host}:{target_port}")

        connection_string = f"postgresql+psycopg2://{pg_user}:{pg_password}@{target_host}:{target_port}/{pg_db}"
        
        engine_args = {
            "pool_pre_ping": True,
            "pool_recycle": 300,
            "connect_args": {
                "keepalives": 1,
                "keepalives_idle": 60,
                "keepalives_interval": 10,
                "keepalives_count": 5
            }
        }
        
        self.vector_store = PGVector(
            connection_string=connection_string,
            embedding_function=self.embeddings,
            collection_name=self.collection_name,
            use_jsonb=True,
            engine_args=engine_args
        )
        return self.vector_store

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.tunnel:
            self.tunnel.stop()
            print("🛑 SSH Tunnel closed.")

# 전역 싱글톤 인스턴스 캐시
_VECTOR_STORE_CACHE = {}

def get_vector_store(collection_name: str = "welding_robotics_manuals") -> PGVector:
    """
    싱글톤 패턴을 적용하여 PGVector 인스턴스를 재사용합니다.
    매번 DB 연결 및 엔진 생성을 하지 않으므로 성능이 대폭 향상됩니다.
    """
    global _VECTOR_STORE_CACHE
    
    if collection_name in _VECTOR_STORE_CACHE:
        return _VECTOR_STORE_CACHE[collection_name]

    # 1. .env에서 기본 DB 정보 가져오기
    pg_user = os.getenv("PGUSER")
    pg_password = os.getenv("PGPASSWORD")
    pg_db = os.getenv("PGDATABASE")
    pg_host = os.getenv("PGHOST")
    pg_port = os.getenv("PGPORT", "5432")

    # 2. SSH 터널링 환경에 따른 분기 처리
    ssh_enabled = os.getenv("SSH_TUNNEL_ENABLED", "false").lower() == "true"
    ssh_local_port = os.getenv("SSH_LOCAL_BIND_PORT")

    if ssh_enabled or ssh_local_port:
        target_host = "127.0.0.1"
        target_port = ssh_local_port if ssh_local_port else pg_port
    else:
        target_host = pg_host
        target_port = pg_port

    connection_string = f"postgresql+psycopg2://{pg_user}:{pg_password}@{target_host}:{target_port}/{pg_db}"
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    engine_args = {
        "pool_pre_ping": True,
        "pool_recycle": 300,
        "connect_args": {
            "keepalives": 1,
            "keepalives_idle": 60,
            "keepalives_interval": 10,
            "keepalives_count": 5
        }
    }

    store = PGVector(
        connection_string=connection_string,
        embedding_function=embeddings,
        collection_name=collection_name,
        use_jsonb=True,
        engine_args=engine_args
    )
    
    _VECTOR_STORE_CACHE[collection_name] = store
    print(f"📦 [PGVector] New singleton instance created for: {collection_name}")
    return store
