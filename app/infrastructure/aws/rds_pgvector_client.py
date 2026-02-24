import os
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlsplit

def env_first(*keys: str):
    for key in keys:
        v = os.getenv(key)
        if v is not None and str(v).strip():
            return str(v).strip()
    return None

def parse_bool(value, default=False):
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y", "on"}

def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value

def get_connection_kwargs() -> dict[str, object]:
    # (팀원분 코드 원본 유지) DB 접속 정보 파싱
    alias_map = {
        "host": ["PGHOST", "POSTGRES_HOST", "DB_HOST"],
        "port": ["PGPORT", "POSTGRES_PORT", "DB_PORT"],
        "dbname": ["PGDATABASE", "POSTGRES_DB", "DB_NAME", "DATABASE_NAME"],
        "user": ["PGUSER", "POSTGRES_USER", "DB_USER", "DATABASE_USER"],
        "password": ["PGPASSWORD", "POSTGRES_PASSWORD", "DB_PASSWORD", "DATABASE_PASSWORD"],
    }

    kwargs: dict[str, object] = {}
    for target_key, aliases in alias_map.items():
        for alias in aliases:
            value = os.getenv(alias)
            if value:
                kwargs[target_key] = int(value) if target_key == "port" else value
                break

    required = ("host", "dbname", "user", "password")
    if all(k in kwargs and kwargs[k] for k in required):
        return kwargs

    raise RuntimeError("PostgreSQL connection info not found in .env")

def get_db_host_port_for_tunnel() -> tuple[str, int]:
    host = env_first("PGHOST", "POSTGRES_HOST", "DB_HOST")
    port_raw = env_first("PGPORT", "POSTGRES_PORT", "DB_PORT")
    if host:
        return host, int(port_raw) if port_raw else 5432
    raise RuntimeError("Cannot determine DB host/port for SSH tunnel.")

def get_ssh_tunnel_config():
    ssh_host = env_first("SSH_HOST", "BASTION_HOST", "SSH_TUNNEL_HOST")
    enabled = parse_bool(env_first("SSH_TUNNEL_ENABLED", "USE_SSH_TUNNEL"), default=bool(ssh_host))
    if not enabled:
        return None

    ssh_user = env_first("SSH_USER", "BASTION_USER", "SSH_TUNNEL_USER")
    remote_host, remote_port = get_db_host_port_for_tunnel()
    return {
        "ssh_host": ssh_host,
        "ssh_port": int(env_first("SSH_PORT", "BASTION_PORT", "SSH_TUNNEL_PORT") or "22"),
        "ssh_user": ssh_user,
        "ssh_key_path": env_first("SSH_PRIVATE_KEY_PATH", "SSH_KEY_PATH", "BASTION_KEY_PATH"),
        "local_bind_port": int(env_first("SSH_LOCAL_BIND_PORT")) if env_first("SSH_LOCAL_BIND_PORT") else None,
        "remote_host": env_first("SSH_REMOTE_BIND_HOST") or remote_host,
        "remote_port": int(env_first("SSH_REMOTE_BIND_PORT") or str(remote_port)),
    }

@contextmanager
def open_optional_ssh_tunnel():
    cfg = get_ssh_tunnel_config()
    if not cfg:
        yield None
        return

    import paramiko
    if not hasattr(paramiko, "DSSKey"):
        paramiko.DSSKey = paramiko.RSAKey
    from sshtunnel import SSHTunnelForwarder

    kwargs = {
        "ssh_address_or_host": (cfg["ssh_host"], cfg["ssh_port"]),
        "ssh_username": cfg["ssh_user"],
        "remote_bind_address": (cfg["remote_host"], cfg["remote_port"]),
        "set_keepalive": 30.0,
    }
    if cfg["local_bind_port"]:
        kwargs["local_bind_address"] = ("127.0.0.1", cfg["local_bind_port"])
    if cfg["ssh_key_path"]:
        kwargs["ssh_pkey"] = str(Path(str(cfg["ssh_key_path"])).expanduser())

    server = SSHTunnelForwarder(**kwargs)
    try:
        server.start()
        print(f"🔒 SSH Tunnel opened at 127.0.0.1:{server.local_bind_port}")
        yield {"forward_host": "127.0.0.1", "forward_port": int(server.local_bind_port)}
    finally:
        server.stop()
        print("🔓 SSH Tunnel closed.")

# 💡 [핵심] LangChain용 DB URL 생성기 추가
@contextmanager
def get_pgvector_url():
    """SSH 터널을 열고 LangChain PGVector에 바로 넣을 수 있는 URL을 반환합니다."""
    # 환경변수 로드
    load_env_file(Path(__file__).resolve().parent.parent.parent.parent / ".env")
    
    conn_kwargs = get_connection_kwargs()
    
    with open_optional_ssh_tunnel() as tunnel:
        # 터널이 뚫렸으면 로컬 호스트/포트를 쓰고, 아니면 원래 접속 정보 사용
        host = tunnel["forward_host"] if tunnel else conn_kwargs["host"]
        port = tunnel["forward_port"] if tunnel else conn_kwargs.get("port", 5432)
        
        user = conn_kwargs["user"]
        password = conn_kwargs["password"]
        dbname = conn_kwargs["dbname"]
        
        # LangChain + pgvector 연결용 SQLAlchemy 포맷 (psycopg 사용)
        db_url = f"postgresql+psycopg://{user}:{password}@{host}:{port}/{dbname}"
        
        yield db_url