from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg2
from psycopg2.extras import execute_values

from app.core.config import DATA_DIR
from app.core.database import get_connection_kwargs, open_optional_ssh_tunnel
from app.ingest.chunking import chunk_markdown_document
from app.vectorstore.pgvector_store import PGVectorStoreManager


REGISTRY_TABLE = "admin_pdf_ingest_registry"


def _slugify_filename(filename: str) -> str:
    stem = Path(filename).stem
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")
    return safe or "uploaded_pdf"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _ensure_registry_table() -> None:
    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS {REGISTRY_TABLE} (
        collection_name TEXT NOT NULL,
        source_key TEXT NOT NULL,
        chunk_id TEXT NOT NULL,
        chunk_hash TEXT NOT NULL,
        file_hash TEXT NOT NULL,
        file_name TEXT NOT NULL,
        chunk_index INTEGER NOT NULL,
        s3_pdf_key TEXT,
        s3_md_key TEXT,
        local_md_path TEXT,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        PRIMARY KEY (collection_name, chunk_id)
    );
    """
    create_index_sql = """
    CREATE INDEX IF NOT EXISTS idx_admin_pdf_ingest_registry_source
        ON admin_pdf_ingest_registry (collection_name, source_key);
    """
    with open_optional_ssh_tunnel() as tunnel:
        conn_args = get_connection_kwargs()
        if tunnel:
            conn_args["host"] = tunnel["host"]
            conn_args["port"] = tunnel["port"]
        with psycopg2.connect(**conn_args) as conn:
            with conn.cursor() as cur:
                cur.execute(create_table_sql)
                cur.execute(create_index_sql)
            conn.commit()


def _fetch_existing_registry(collection_name: str, source_key: str) -> dict[str, dict[str, Any]]:
    _ensure_registry_table()
    with open_optional_ssh_tunnel() as tunnel:
        conn_args = get_connection_kwargs()
        if tunnel:
            conn_args["host"] = tunnel["host"]
            conn_args["port"] = tunnel["port"]
        with psycopg2.connect(**conn_args) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT chunk_id, chunk_hash, file_hash
                    FROM {REGISTRY_TABLE}
                    WHERE collection_name = %s AND source_key = %s
                    """,
                    (collection_name, source_key),
                )
                rows = cur.fetchall()
    return {row[0]: {"chunk_hash": row[1], "file_hash": row[2]} for row in rows}


def _delete_registry_rows(collection_name: str, chunk_ids: list[str]) -> None:
    if not chunk_ids:
        return
    with open_optional_ssh_tunnel() as tunnel:
        conn_args = get_connection_kwargs()
        if tunnel:
            conn_args["host"] = tunnel["host"]
            conn_args["port"] = tunnel["port"]
        with psycopg2.connect(**conn_args) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"DELETE FROM {REGISTRY_TABLE} WHERE collection_name = %s AND chunk_id = ANY(%s)",
                    (collection_name, chunk_ids),
                )
            conn.commit()


def _upsert_registry_rows(rows: list[tuple[Any, ...]]) -> None:
    if not rows:
        return
    with open_optional_ssh_tunnel() as tunnel:
        conn_args = get_connection_kwargs()
        if tunnel:
            conn_args["host"] = tunnel["host"]
            conn_args["port"] = tunnel["port"]
        with psycopg2.connect(**conn_args) as conn:
            with conn.cursor() as cur:
                execute_values(
                    cur,
                    f"""
                    INSERT INTO {REGISTRY_TABLE} (
                        collection_name, source_key, chunk_id, chunk_hash, file_hash,
                        file_name, chunk_index, s3_pdf_key, s3_md_key, local_md_path
                    ) VALUES %s
                    ON CONFLICT (collection_name, chunk_id) DO UPDATE SET
                        source_key = EXCLUDED.source_key,
                        chunk_hash = EXCLUDED.chunk_hash,
                        file_hash = EXCLUDED.file_hash,
                        file_name = EXCLUDED.file_name,
                        chunk_index = EXCLUDED.chunk_index,
                        s3_pdf_key = EXCLUDED.s3_pdf_key,
                        s3_md_key = EXCLUDED.s3_md_key,
                        local_md_path = EXCLUDED.local_md_path,
                        updated_at = NOW()
                    """,
                    rows,
                )
            conn.commit()


def _write_local_markdown(markdown_text: str, original_filename: str, file_hash: str) -> str:
    out_dir = DATA_DIR / "processed" / "uploads_md"
    out_dir.mkdir(parents=True, exist_ok=True)
    base = _slugify_filename(original_filename)
    md_path = out_dir / f"{base}__{file_hash[:8]}.md"
    md_path.write_text(markdown_text, encoding="utf-8")
    return str(md_path)


def _build_legacy_cmetadata(doc: Any, original_filename: str) -> dict[str, Any]:
    """
    Match legacy cmetadata style as closely as possible:
    {
      "Header 1": ...,
      "Header 2": ...,
      "Header 3": ...,
      "source_file": "...md"
    }
    """
    src_md_name = f"{Path(original_filename).stem}.md"
    raw = dict(getattr(doc, "metadata", {}) or {})

    header1 = raw.get("Header 1") or ""
    header2 = raw.get("Header 2") or ""
    header3 = raw.get("Header 3") or ""

    if not (header1 or header2 or header3):
        chapter_path = str(raw.get("chapter_path", ""))
        parts = [p.strip() for p in chapter_path.split(" > ") if p.strip()]
        if len(parts) > 0:
            header1 = parts[0]
        if len(parts) > 1:
            header2 = parts[1]
        if len(parts) > 2:
            header3 = parts[2]

    return {
        "Header 1": header1,
        "Header 2": header2,
        "Header 3": header3,
        "source_file": src_md_name,
    }


def _update_vector_audit_columns(
    *,
    collection_name: str,
    chunk_ids: list[str],
    creator: str,
    created_at: datetime,
) -> None:
    if not chunk_ids:
        return

    creator_value = (creator or "").strip() or "admin"
    with open_optional_ssh_tunnel() as tunnel:
        conn_args = get_connection_kwargs()
        if tunnel:
            conn_args["host"] = tunnel["host"]
            conn_args["port"] = tunnel["port"]
        with psycopg2.connect(**conn_args) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE langchain_pg_embedding e
                    SET created_at = %s,
                        creator = %s
                    FROM langchain_pg_collection c
                    WHERE e.collection_id = c.uuid
                      AND c.name = %s
                      AND e.custom_id = ANY(%s)
                    """,
                    (created_at, creator_value, collection_name, chunk_ids),
                )
            conn.commit()


def incremental_embed_markdown_to_pgvector(
    *,
    markdown_text: str,
    file_bytes: bytes,
    original_filename: str,
    creator: str = "admin",
    collection_name: str = "welding_robotics_manuals",
    s3_pdf_key: str | None = None,
    s3_md_key: str | None = None,
) -> dict[str, Any]:
    """
    Incrementally upsert parsed markdown chunks into PGVector.

    Behavior:
    - Parses markdown into chunks
    - Computes deterministic chunk IDs
    - Inserts only new chunks
    - Deletes stale chunks for the same uploaded filename
    """
    file_hash = _sha256_bytes(file_bytes)
    source_key = original_filename.strip()
    local_md_path = _write_local_markdown(markdown_text, original_filename, file_hash)
    upload_time = datetime.now(timezone.utc)

    chunks = chunk_markdown_document(markdown_text, source_path=local_md_path)
    current_chunk_ids: list[str] = []
    current_rows: list[tuple[Any, ...]] = []
    docs_by_chunk_id: dict[str, Any] = {}

    for idx, doc in enumerate(chunks):
        raw_meta = dict(getattr(doc, "metadata", {}) or {})
        chapter_path = str(raw_meta.get("chapter_path", ""))
        cmetadata = _build_legacy_cmetadata(doc, original_filename)
        chunk_hash = _sha256_text(f"{chapter_path}\n{doc.page_content}")
        chunk_id = _sha256_text(f"{source_key}\n{chunk_hash}")
        current_chunk_ids.append(chunk_id)
        doc.metadata = cmetadata
        docs_by_chunk_id[chunk_id] = doc
        current_rows.append(
            (
                collection_name,
                source_key,
                chunk_id,
                chunk_hash,
                file_hash,
                original_filename,
                idx,
                s3_pdf_key,
                s3_md_key,
                local_md_path,
            )
        )

    existing = _fetch_existing_registry(collection_name, source_key)
    existing_ids = set(existing.keys())
    current_ids = set(current_chunk_ids)

    new_ids = [cid for cid in current_chunk_ids if cid not in existing_ids]
    stale_ids = [cid for cid in existing_ids if cid not in current_ids]
    skipped_count = len(current_chunk_ids) - len(new_ids)

    with PGVectorStoreManager(collection_name=collection_name) as vector_store:
        if stale_ids:
            vector_store.delete(ids=stale_ids, collection_only=True)
        if new_ids:
            vector_store.add_documents([docs_by_chunk_id[cid] for cid in new_ids], ids=new_ids)
            _update_vector_audit_columns(
                collection_name=collection_name,
                chunk_ids=new_ids,
                creator=creator,
                created_at=upload_time,
            )

    if stale_ids:
        _delete_registry_rows(collection_name, stale_ids)
    _upsert_registry_rows(current_rows)

    bm25_result: dict[str, Any] = {
        "bm25_cache_updated": False,
        "bm25_status": "not_attempted",
        "bm25_total_docs": None,
        "bm25_source_docs": len(current_chunk_ids),
        "bm25_prev_source_docs": None,
    }
    try:
        from app.rag.retriever import update_bm25_cache_for_uploaded_source

        source_file = f"{Path(original_filename).stem}.md"
        ordered_docs = [docs_by_chunk_id[cid] for cid in current_chunk_ids]
        bm25_result = update_bm25_cache_for_uploaded_source(
            ordered_docs,
            source_file=source_file,
        )
    except Exception as e:
        bm25_result = {
            "bm25_cache_updated": False,
            "bm25_status": f"error:{e}",
            "bm25_total_docs": None,
            "bm25_source_docs": len(current_chunk_ids),
            "bm25_prev_source_docs": None,
        }

    unchanged = bool(existing_ids) and not new_ids and not stale_ids
    return {
        "message_suffix": "No new chunks detected (already indexed)." if unchanged else "Incremental embedding completed.",
        "local_md_path": local_md_path,
        "file_hash": file_hash,
        "total_chunks_parsed": len(current_chunk_ids),
        "db_chunks_inserted": len(new_ids),
        "db_chunks_skipped": skipped_count,
        "db_chunks_deleted": len(stale_ids),
        **bm25_result,
    }
