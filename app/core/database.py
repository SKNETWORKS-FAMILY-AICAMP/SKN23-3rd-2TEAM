import os
import psycopg2
import bcrypt
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
from sshtunnel import SSHTunnelForwarder
from dotenv import load_dotenv  # 🌟 추가: .env 로드용

# 프로젝트 루트의 .env 파일을 명시적으로 로드합니다.
load_dotenv()

# ==========================================
# 0. Infrastructure & Tunneling Utils
# ==========================================
def env_first(*keys: str):
    """여러 환경 변수 중 가장 먼저 매칭되는 값을 반환합니다."""
    for key in keys:
        v = os.getenv(key)
        if v is not None and str(v).strip():
            return str(v).strip()
    return None

def parse_bool(value, default=False):
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y", "on"}

import socket

def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0

@contextmanager
def open_optional_ssh_tunnel():
    """SSH 터널링을 컨텍스트 매니저로 제공합니다."""
    ssh_host = env_first("SSH_HOST", "BASTION_HOST")
    # SSH_TUNNEL_ENABLED가 명시되지 않아도 SSH_HOST가 있으면 True로 간주합니다.
    enabled = parse_bool(env_first("SSH_TUNNEL_ENABLED", "USE_SSH_TUNNEL"), default=bool(ssh_host))
    
    if not enabled:
        yield None
        return

    local_port = int(env_first("SSH_LOCAL_BIND_PORT") or "15432")
    
    # 🌟 핵심 수정: 백그라운드 스크립트(run_tunnel.py)가 이미 터널을 뚫어두었다면 바로 포트 재사용!
    if is_port_in_use(local_port):
        yield {"host": "127.0.0.1", "port": local_port}
        return

    # 만약 백그라운드 로직이 없다면 직접 일회성 터널 개최
    try:
        ssh_user = env_first("SSH_USER", "BASTION_USER")
        ssh_key_path = env_first("SSH_PRIVATE_KEY_PATH", "SSH_KEY_PATH")
        target_host = env_first("PGHOST", "DB_HOST")
        target_port = int(env_first("PGPORT", "DB_PORT") or "5432")

        server = SSHTunnelForwarder(
            (ssh_host, int(env_first("SSH_PORT") or "22")),
            ssh_username=ssh_user,
            ssh_pkey=str(Path(ssh_key_path).expanduser()) if ssh_key_path else None,
            remote_bind_address=(target_host, target_port),
            local_bind_address=('127.0.0.1', local_port),
            set_keepalive=30.0
        )
        
        server.start()
        yield {"host": "127.0.0.1", "port": server.local_bind_port}
    except Exception as e:
        print(f"❌ DB SSH Tunnel Failed: {e}")
        yield None
    finally:
        if 'server' in locals() and server.is_active:
            server.stop()

def get_connection_kwargs():
    """DB 연결을 위한 기본 파라미터를 구성합니다."""
    return {
        "dbname": env_first("PGDATABASE", "DB_NAME", "chatbot_db"),
        "user": env_first("PGUSER", "DB_USER", "postgres"),
        "password": env_first("PGPASSWORD", "DB_PASSWORD", "password"),
        "host": env_first("PGHOST", "DB_HOST", "localhost"),
        "port": int(env_first("PGPORT", "DB_PORT") or "5432"),
        "connect_timeout": 5
    }

# ==========================================
# 1. Password Management
# ==========================================
def hash_password(password: str) -> str:
    pwd_bytes = password.encode('utf-8')
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(pwd_bytes, salt).decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))
    except Exception:
        return False

# ==========================================
# 2. User & Chat Logging
# ==========================================
def create_user(username: str, password: str, name: str = "", role: str = "user") -> bool:
    hashed = hash_password(password)
    try:
        with open_optional_ssh_tunnel() as tunnel:
            conn_args = get_connection_kwargs()
            if tunnel:
                conn_args["host"] = tunnel["host"]
                conn_args["port"] = tunnel["port"]
            
            with psycopg2.connect(**conn_args) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO users (username, password_hash, name, role) VALUES (%s, %s, %s, %s)",
                        (username, hashed, name, role)
                    )
                conn.commit()
        return True
    except Exception as e:
        print(f"❌ User Creation Failed: {e}")
        return False

def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    try:
        with open_optional_ssh_tunnel() as tunnel:
            conn_args = get_connection_kwargs()
            if tunnel:
                conn_args["host"] = tunnel["host"]
                conn_args["port"] = tunnel["port"]
            
            with psycopg2.connect(**conn_args) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT id, username, password_hash, role, name FROM users WHERE username = %s", (username,))
                    row = cur.fetchone()
                    if row:
                        return {
                            "id": str(row[0]), 
                            "username": row[1], 
                            "password": row[2], 
                            "role": row[3],
                            "name": row[4] or row[1] # Fallback to username if name is empty
                        }
    except Exception as e:
        print(f"❌ User Query Failed: {e}")
    return None

def log_chat_interaction(
    user_id: str, 
    thread_id: str, 
    query: str, 
    response: str, 
    latency: float = 0.0, 
    reranker_score: float | None = None, 
    verifier_passed: bool | None = None,
    generation_score: int | None = None,
    retrieval_total_chunks: int | None = None,
    retrieval_relevant_chunks: int | None = None,
    retrieval_is_answerable: bool | None = None,
    eval_reason: str | None = None,
    context: str | None = None,
    generation_model: str | None = None,
    evaluation_model: str | None = None
):
    try:
        with open_optional_ssh_tunnel() as tunnel:
            conn_args = get_connection_kwargs()
            if tunnel:
                conn_args["host"] = tunnel["host"]
                conn_args["port"] = tunnel["port"]
            
            with psycopg2.connect(**conn_args) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO chat_logs 
                        (user_id, thread_id, query, response, latency, reranker_score, verifier_passed,
                         generation_score, retrieval_total_chunks, retrieval_relevant_chunks, retrieval_is_answerable, eval_reason, context, generation_model, evaluation_model) 
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (user_id, thread_id, query, response, latency, reranker_score, verifier_passed,
                         generation_score, retrieval_total_chunks, retrieval_relevant_chunks, retrieval_is_answerable, eval_reason, context, generation_model, evaluation_model)
                    )
                conn.commit()
    except Exception as e:
        print(f"❌ Chat Logging Failed: {e}")

def get_chat_logs(limit: int = 50) -> List[Dict[str, Any]]:
    try:
        with open_optional_ssh_tunnel() as tunnel:
            conn_args = get_connection_kwargs()
            if tunnel:
                conn_args["host"] = tunnel["host"]
                conn_args["port"] = tunnel["port"]
            
            with psycopg2.connect(**conn_args) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT c.id, u.username, c.query, c.response, c.latency, 
                               c.reranker_score, c.verifier_passed, c.created_at,
                               c.generation_score, c.retrieval_total_chunks, c.retrieval_relevant_chunks,
                               c.retrieval_is_answerable, c.eval_reason, c.context,
                               c.generation_model, c.evaluation_model
                        FROM chat_logs c 
                        JOIN users u ON c.user_id = u.id 
                        ORDER BY c.created_at DESC 
                        LIMIT %s
                    """, (limit,))
                    rows = cur.fetchall()
                    return [
                        {
                            "id": r[0], "username": r[1], "query": r[2], "response": r[3], 
                            "latency": r[4], "reranker_score": r[5], "verifier_passed": r[6], "timestamp": r[7],
                            "generation_score": r[8], "retrieval_total_chunks": r[9], 
                            "retrieval_relevant_chunks": r[10], "retrieval_is_answerable": r[11],
                            "eval_reason": r[12], "context": r[13],
                            "generation_model": r[14], "evaluation_model": r[15]
                        } 
                        for r in rows
                    ]
    except Exception as e:
        print(f"❌ Log Query Failed: {e}")
        raise e

# ==========================================
# 3. Admin Settings & Evaluation Prompts
# ==========================================
from app.core.prompts import DEFAULT_EVAL_GEN_PROMPT, DEFAULT_EVAL_RET_PROMPT

def init_admin_settings_table():
    try:
        with open_optional_ssh_tunnel() as tunnel:
            conn_args = get_connection_kwargs()
            if tunnel:
                conn_args["host"] = tunnel["host"]
                conn_args["port"] = tunnel["port"]
            
            with psycopg2.connect(**conn_args) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS admin_settings (
                            id INT PRIMARY KEY,
                            evaluation_model VARCHAR(255) DEFAULT 'gpt-4o',
                            generation_prompt TEXT,
                            retrieval_prompt TEXT,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
                    
                    cur.execute("SELECT id FROM admin_settings WHERE id = 1")
                    if not cur.fetchone():
                        cur.execute(
                            """
                            INSERT INTO admin_settings (id, evaluation_model, generation_prompt, retrieval_prompt) 
                            VALUES (1, %s, %s, %s)
                            """,
                            ('gpt-4o', DEFAULT_EVAL_GEN_PROMPT, DEFAULT_EVAL_RET_PROMPT)
                        )
                conn.commit()
    except Exception as e:
        print(f"❌ Admin Settings Init Failed: {e}")
