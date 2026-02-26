import psycopg2
import postgresql_connection as pc
from pathlib import Path
import os
from openai import OpenAI

# ==============================
# 1️⃣ 환경변수 로드
# ==============================
pc.load_env_file(Path(__file__).resolve().parent / ".env")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY 환경변수 없음")

client = OpenAI(api_key=api_key)

conn_kwargs = pc.get_connection_kwargs()


# ==============================
# 2️⃣ Overlap Chunk 함수
# ==============================
def chunk_text(text, chunk_size=800, overlap=150):
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


# ==============================
# 3️⃣ DB 연결
# ==============================
with pc.open_optional_ssh_tunnel() as tunnel:

    effective_conn_kwargs = dict(conn_kwargs)

    if tunnel:
        effective_conn_kwargs["host"] = tunnel["forward_host"]
        effective_conn_kwargs["port"] = tunnel["forward_port"]

    with psycopg2.connect(**effective_conn_kwargs) as conn:
        with conn.cursor() as cur:

            metadata_dir = "./metadata"
            content_dir = "./content"

            for meta_file in os.listdir(metadata_dir)[:5]:

                if not meta_file.endswith("_meta.json"):
                    continue

                base_name = meta_file.replace("_meta.json", "")
                md_path = os.path.join(content_dir, f"{base_name}.md")

                if not os.path.exists(md_path):
                    print("MD 없음:", md_path)
                    continue

                print(f"📄 처리중: {base_name}")

                with open(md_path, "r", encoding="utf-8") as f:
                    full_text = f.read()

                chunks = chunk_text(full_text)

                if not chunks:
                    continue

                # ==============================
                # 🔥 4️⃣ Batch Embedding (핵심)
                # ==============================
                response = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=chunks
                )

                embeddings = [item.embedding for item in response.data]

                # ==============================
                # 🔥 5️⃣ Batch Insert
                # ==============================
                records = []

                for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                    records.append((
                        base_name,
                        idx,
                        chunk,
                        embedding
                    ))

                cur.executemany("""
                    INSERT INTO document_chunks
                    (doc_id, chunk_index, content, embedding)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (doc_id, chunk_index)
                    DO UPDATE SET
                        content = EXCLUDED.content,
                        embedding = EXCLUDED.embedding
                """, records)

        conn.commit()

print("✅ RAG용 chunk + embedding 저장 완료 (최적화 버전)")