import os
import sys
import asyncio
import contextlib

from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool


# Windows + psycopg async compatibility:
# Force SelectorEventLoop policy early so libraries imported later inherit it when possible.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


class AsyncCompatPostgresSaver(PostgresSaver):
    """Thread-offloaded async wrappers for PostgresSaver on Windows."""

    async def aget_tuple(self, config):
        return await asyncio.to_thread(self.get_tuple, config)

    async def aget(self, config):
        return await asyncio.to_thread(self.get, config)

    async def aput(self, config, checkpoint, metadata, new_versions):
        return await asyncio.to_thread(self.put, config, checkpoint, metadata, new_versions)

    async def aput_writes(self, config, writes, task_id, task_path=""):
        return await asyncio.to_thread(self.put_writes, config, writes, task_id, task_path)

    async def adelete_thread(self, thread_id):
        return await asyncio.to_thread(self.delete_thread, thread_id)

    async def alist(self, config, *, filter=None, before=None, limit=None):
        items = await asyncio.to_thread(
            lambda: list(self.list(config, filter=filter, before=before, limit=limit))
        )
        for item in items:
            yield item


def _get_conn_info():
    """Build PostgreSQL conninfo from env vars."""
    host = os.getenv("PGHOST", "localhost")
    user = os.getenv("PGUSER", "postgres")
    pw = os.getenv("PGPASSWORD", "password")
    db = os.getenv("PGDATABASE", "chatbot_db")
    port = os.getenv("PGPORT", "5432")

    # Route through local SSH tunnel when enabled.
    if os.getenv("SSH_TUNNEL_ENABLED", "false").lower() == "true":
        host = "127.0.0.1"
        port = os.getenv("SSH_LOCAL_BIND_PORT", "15432")

    keepalives = "keepalives=1 keepalives_idle=60 keepalives_interval=10 keepalives_count=5 connect_timeout=5"
    return f"host={host} user={user} password={pw} dbname={db} port={port} {keepalives}"


def get_memory_saver():
    """
    Historical name kept for compatibility.
    Returns a Postgres-backed saver with a sync connection pool.
    """
    conninfo = _get_conn_info()
    pool = ConnectionPool(conninfo, max_size=10, min_size=1, max_lifetime=300)
    checkpointer = PostgresSaver(pool)
    checkpointer.setup()
    return checkpointer


@contextlib.asynccontextmanager
async def get_async_postgres_saver():
    """
    Async context manager for LangGraph checkpointer.

    On Windows, psycopg async pool may fail under ProactorEventLoop.
    We use sync ConnectionPool + PostgresSaver (which still exposes async methods)
    to avoid that runtime incompatibility.
    """
    conninfo = _get_conn_info()

    if sys.platform == "win32":
        pool = ConnectionPool(conninfo, max_size=10, min_size=1, max_lifetime=300)
        try:
            checkpointer = AsyncCompatPostgresSaver(pool)
            await asyncio.to_thread(checkpointer.setup)
            yield checkpointer
        finally:
            pool.close()
        return

    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from psycopg_pool import AsyncConnectionPool

    async with AsyncConnectionPool(conninfo, max_size=10, max_lifetime=300) as pool:
        async with pool.connection() as conn:
            checkpointer = AsyncPostgresSaver(conn)
            await checkpointer.setup()
            yield checkpointer
