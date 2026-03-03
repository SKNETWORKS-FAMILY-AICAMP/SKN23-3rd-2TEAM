from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone, date
from pathlib import Path
from typing import Any, Optional

import psycopg2
from psycopg2.extras import execute_values

from app.core.config import DATA_DIR
from app.core.database import get_connection_kwargs, open_optional_ssh_tunnel
from app.ingest.chunking import chunk_markdown_document
from app.vectorstore.pgvector_store import PGVectorStoreManager


REGISTRY_TABLE = "admin_pdf_ingest_registry"
USE_YN_COLUMN_READY = False


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


def _ensure_use_yn_column() -> None:
    """
    langchain_pg_embedding.use_yn 컬럼과 기본값을 보장합니다.
    대량 데이터 백필은 운영 SQL로 별도 수행합니다.
    """
    global USE_YN_COLUMN_READY
    if USE_YN_COLUMN_READY:
        return

    with open_optional_ssh_tunnel() as tunnel:
        conn_args = get_connection_kwargs()
        if tunnel:
            conn_args["host"] = tunnel["host"]
            conn_args["port"] = tunnel["port"]
        with psycopg2.connect(**conn_args) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    ALTER TABLE langchain_pg_embedding
                    ADD COLUMN IF NOT EXISTS use_yn CHAR(1);
                    """
                )
                cur.execute(
                    """
                    ALTER TABLE langchain_pg_embedding
                    ALTER COLUMN use_yn SET DEFAULT 'Y';
                    """
                )
            conn.commit()
    USE_YN_COLUMN_READY = True


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


def get_all_registry_documents(collection_name: str = "welding_robotics_manuals") -> list[dict[str, Any]]:
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
                    SELECT source_key, COUNT(chunk_id) as chunk_count, 
                           MAX(created_at) as created_at, 
                           MAX(s3_pdf_key) as s3_pdf_key, MAX(s3_md_key) as s3_md_key
                    FROM {REGISTRY_TABLE}
                    WHERE collection_name = %s
                    GROUP BY source_key
                    ORDER BY created_at DESC
                    """,
                    (collection_name,)
                )
                rows = cur.fetchall()
                # To get creator, we can either join or just fetch from PGVector metadata, but a simpler way is to fetch creator from langchain_pg_embedding
                # For now let's just return the basic info and we'll fetch creator directly in a joined query
                
                cur.execute(
                    f"""
                    SELECT r.source_key, r.chunk_count, r.created_at, r.s3_pdf_key, r.s3_md_key, e.creator
                    FROM (
                        SELECT source_key, COUNT(chunk_id) as chunk_count, 
                               MAX(created_at) as created_at, 
                               MAX(s3_pdf_key) as s3_pdf_key, MAX(s3_md_key) as s3_md_key,
                               MAX(chunk_id) as any_chunk_id
                        FROM {REGISTRY_TABLE}
                        WHERE collection_name = %s
                        GROUP BY source_key
                    ) r
                    LEFT JOIN langchain_pg_embedding e ON e.custom_id = r.any_chunk_id
                    ORDER BY r.created_at DESC
                    """,
                    (collection_name,)
                )
                rows_with_creator = cur.fetchall()
                
    return [
        {
            "source_key": row[0],
            "chunk_count": row[1],
            "created_at": row[2],
            "s3_pdf_key": row[3],
            "s3_md_key": row[4],
            "creator": row[5] or "admin"
        }
        for row in rows_with_creator
    ]


def search_embedding_sources(
    *,
    collection_name: str = "welding_robotics_manuals",
    file_name: Optional[str] = None,
    creator: Optional[str] = None,
    uploaded_date: Optional[str] = None,
    use_yn: str = "ALL",
    limit: int = 100,
) -> list[dict[str, Any]]:
    """
    source_key(파일) 단위로 임베딩 상태를 조회합니다.
    검색 조건: 파일명, creator 컬럼, created_at 컬럼 날짜(YYYY-MM-DD), use_yn(Y/N/ALL)
    """
    _ensure_use_yn_column()
    limit = max(1, min(int(limit or 200), 1000))

    source_expr = """
        COALESCE(
            NULLIF(BTRIM(e.cmetadata->>'source_file'), ''),
            NULLIF(BTRIM(e.cmetadata->>'source_key'), ''),
            NULLIF(BTRIM(e.cmetadata->>'source'), ''),
            e.custom_id
        )
    """

    filters: list[str] = ["c.name = %s"]
    params: list[Any] = [collection_name]

    if file_name and file_name.strip():
        filters.append(f"LOWER({source_expr}) LIKE LOWER(%s)")
        params.append(f"%{file_name.strip()}%")

    if creator and creator.strip():
        filters.append("LOWER(COALESCE(NULLIF(BTRIM(e.creator), ''), '')) LIKE LOWER(%s)")
        params.append(f"%{creator.strip()}%")

    normalized_use_yn = (use_yn or "ALL").strip().upper()
    if normalized_use_yn in {"Y", "N"}:
        filters.append("COALESCE(e.use_yn, 'Y') = %s")
        params.append(normalized_use_yn)

    upload_date_obj: Optional[date] = None
    if uploaded_date and uploaded_date.strip():
        upload_date_obj = datetime.strptime(uploaded_date.strip(), "%Y-%m-%d").date()
        filters.append("DATE(e.created_at) = %s")
        params.append(upload_date_obj)

    where_sql = " AND ".join(filters)

    query = f"""
        SELECT
            {source_expr} AS source_key,
            COUNT(*) AS chunk_count,
            SUM(CASE WHEN COALESCE(e.use_yn, 'Y') = 'Y' THEN 1 ELSE 0 END) AS active_chunks,
            SUM(CASE WHEN COALESCE(e.use_yn, 'Y') = 'N' THEN 1 ELSE 0 END) AS inactive_chunks,
            COALESCE(MAX(NULLIF(BTRIM(e.creator), '')), 'admin') AS creator,
            MAX(e.created_at) AS uploaded_at,
            CASE
                WHEN SUM(CASE WHEN COALESCE(e.use_yn, 'Y') = 'Y' THEN 1 ELSE 0 END) = 0 THEN 'N'
                ELSE 'Y'
            END AS use_yn
        FROM langchain_pg_embedding e
        JOIN langchain_pg_collection c
          ON e.collection_id = c.uuid
        WHERE {where_sql}
        GROUP BY 1
        ORDER BY uploaded_at DESC NULLS LAST, source_key ASC
        LIMIT %s
    """

    with open_optional_ssh_tunnel() as tunnel:
        conn_args = get_connection_kwargs()
        if tunnel:
            conn_args["host"] = tunnel["host"]
            conn_args["port"] = tunnel["port"]
        with psycopg2.connect(**conn_args) as conn:
            with conn.cursor() as cur:
                cur.execute(query, [*params, limit])
                rows = cur.fetchall()

    return [
        {
            "source_key": row[0],
            "chunk_count": int(row[1] or 0),
            "active_chunks": int(row[2] or 0),
            "inactive_chunks": int(row[3] or 0),
            "creator": row[4] or "admin",
            "uploaded_at": row[5],
            "use_yn": row[6] or "Y",
        }
        for row in rows
    ]


def set_use_yn_by_sources(
    *,
    source_keys: list[str],
    use_yn: str,
    collection_name: str = "welding_robotics_manuals",
) -> dict[str, Any]:
    """
    source_key 단위로 langchain_pg_embedding.use_yn 상태를 일괄 업데이트합니다.
    """
    _ensure_use_yn_column()

    normalized = (use_yn or "").strip().upper()
    if normalized not in {"Y", "N"}:
        raise ValueError("use_yn 값은 'Y' 또는 'N' 이어야 합니다.")

    clean_keys = [k.strip() for k in (source_keys or []) if isinstance(k, str) and k.strip()]
    if not clean_keys:
        return {"updated_chunks": 0, "target_sources": 0, "use_yn": normalized}

    source_expr = """
        COALESCE(
            NULLIF(BTRIM(e.cmetadata->>'source_file'), ''),
            NULLIF(BTRIM(e.cmetadata->>'source_key'), ''),
            NULLIF(BTRIM(e.cmetadata->>'source'), ''),
            e.custom_id
        )
    """

    with open_optional_ssh_tunnel() as tunnel:
        conn_args = get_connection_kwargs()
        if tunnel:
            conn_args["host"] = tunnel["host"]
            conn_args["port"] = tunnel["port"]
        with psycopg2.connect(**conn_args) as conn:
            with conn.cursor() as cur:
                update_sql = (
                    """
                    UPDATE langchain_pg_embedding e
                    SET use_yn = %s,
                        cmetadata = jsonb_set(
                            COALESCE(e.cmetadata, '{}'::jsonb),
                            '{use_yn}',
                            to_jsonb(%s::text),
                            true
                        )
                    FROM langchain_pg_collection c
                    WHERE e.collection_id = c.uuid
                      AND c.name = %s
                      AND """
                    + source_expr
                    + """ = ANY(%s)
                    """
                )
                cur.execute(
                    update_sql,
                    (normalized, normalized, collection_name, clean_keys),
                )
                updated_chunks = cur.rowcount
            conn.commit()

    return {
        "updated_chunks": int(updated_chunks or 0),
        "target_sources": len(clean_keys),
        "use_yn": normalized,
    }


def delete_registry_documents_by_sources(source_keys: list[str], collection_name: str = "welding_robotics_manuals") -> dict[str, Any]:
    if not source_keys:
        return {"deleted_chunks": 0, "s3_keys_to_delete": []}
        
    _ensure_registry_table()
    s3_keys_to_delete = []
    chunk_ids_to_delete = []
    
    with open_optional_ssh_tunnel() as tunnel:
        conn_args = get_connection_kwargs()
        if tunnel:
            conn_args["host"] = tunnel["host"]
            conn_args["port"] = tunnel["port"]
        with psycopg2.connect(**conn_args) as conn:
            with conn.cursor() as cur:
                # 1. Fetch chunk_ids and s3 keys
                cur.execute(
                    f"""
                    SELECT chunk_id, s3_pdf_key, s3_md_key 
                    FROM {REGISTRY_TABLE}
                    WHERE collection_name = %s AND source_key = ANY(%s)
                    """,
                    (collection_name, source_keys)
                )
                rows = cur.fetchall()
                for row in rows:
                    chunk_ids_to_delete.append(row[0])
                    if row[1] and row[1] not in s3_keys_to_delete:
                        s3_keys_to_delete.append(row[1])
                    if row[2] and row[2] not in s3_keys_to_delete:
                        s3_keys_to_delete.append(row[2])
                    
                    # Also reconstruct json key if pdf key exists
                    if row[1]:
                        s3_json_key = row[1] + '.json'
                        if s3_json_key not in s3_keys_to_delete:
                            s3_keys_to_delete.append(s3_json_key)

    if chunk_ids_to_delete:
        with PGVectorStoreManager(collection_name=collection_name) as vector_store:
            vector_store.delete(ids=chunk_ids_to_delete, collection_only=True)
        _delete_registry_rows(collection_name, chunk_ids_to_delete)
        
    return {
        "deleted_chunks": len(chunk_ids_to_delete),
        "s3_keys_to_delete": s3_keys_to_delete
    }

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
                        creator = %s,
                        cmetadata = jsonb_set(
                            COALESCE(e.cmetadata, '{}'::jsonb),
                            '{use_yn}',
                            to_jsonb('Y'::text),
                            true
                        )
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
    file_metadata_json: str | None = None,
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
        
        if file_metadata_json:
            import json
            try:
                parsed_meta = json.loads(file_metadata_json)
                for k, v in parsed_meta.items():
                    if isinstance(v, list):
                        cmetadata[k] = ", ".join(map(str, v))
                    elif isinstance(v, (str, int, float, bool)):
                        cmetadata[k] = v
                    else:
                        cmetadata[k] = str(v)
            except json.JSONDecodeError:
                pass
        cmetadata["use_yn"] = "Y"
                
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
