import sys
import os
import json
import glob

# 프로젝트 루트를 sys.path에 추가하여 app 모듈을 불러올 수 있게 설정
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from app.vectorstore.pgvector_store import PGVectorStoreManager

def load_and_ingest_data():
    print("🔄 [System] 데이터 인제스트 파이프라인 시작 (pgvector 전용)...")
    
    # 데이터 경로 설정 (OS 호환성 위해 os.path.join 활용)
    processed_dir = os.path.join(PROJECT_ROOT, "data", "processed")
    md_dir = os.path.join(processed_dir, "md_files")
    json_dir = os.path.join(processed_dir, "json_files")
    
    if not os.path.exists(md_dir):
        print(f"❌ 경로를 찾을 수 없습니다: {md_dir}")
        return

    # 1. 스플리터 설정
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    # 1000자 단위, overlap 150자
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

    md_files = glob.glob(os.path.join(md_dir, "*.md"))
    print(f"✅ 총 {len(md_files)}개의 마크다운 파일을 발견했습니다.")

    total_chunks = 0
    
    # 2. PGVectorStoreManager를 사용하여 SSH 터널링 자동 관리
    print("🔄 SSH 터널링 및 DB 연결 시도 중...")
    try:
        with PGVectorStoreManager(collection_name="welding_robotics_manuals") as vector_store:
            print("✅ DB 연결 완료. 파일 처리를 시작합니다.")
            for md_path in md_files:
                filename = os.path.basename(md_path)
                base_name = os.path.splitext(filename)[0]
                json_path = os.path.join(json_dir, f"{base_name}.json")
                
                # 메타데이터 로드
                metadata = {"source_file": filename}
                if os.path.exists(json_path):
                    try:
                        with open(json_path, 'r', encoding='utf-8') as f:
                            json_data = json.load(f)
                            if isinstance(json_data, dict):
                                metadata.update(json_data)
                    except Exception as e:
                        print(f"⚠️ {base_name}.json 읽기 실패: {e}")

                # 마크다운 로드 및 분할
                with open(md_path, 'r', encoding='utf-8') as f:
                    md_text = f.read()

                print(f"🔄 [{filename}] 분할 중...")
                # 1차 분할: 헤더 기준
                md_header_splits = markdown_splitter.split_text(md_text)
                # 2차 분할: 글자 수 기준
                splits = text_splitter.split_documents(md_header_splits)
                print(f"📦 [{filename}] {len(splits)}개 청크 생성됨.")
                
                # 메타데이터 병합
                for split in splits:
                    split.metadata.update(metadata)
                    
                if splits:
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            vector_store.add_documents(splits)
                            total_chunks += len(splits)
                            print(f"✅ [{filename}] -> {len(splits)}개 청크 적재 완료")
                            break
                        except Exception as e:
                            if "rate_limit_exceeded" in str(e).lower() and attempt < max_retries - 1:
                                print(f"⚠️ [{filename}] Rate limit hit, retrying in 5s... (Attempt {attempt+1}/{max_retries})")
                                import time
                                time.sleep(5)
                            else:
                                print(f"❌ [{filename}] 적재 중 에러: {e}")
                                break
                    # 파일 간 짧은 휴식 (Rate limit 방지)
                    import time
                    time.sleep(0.5)

    except Exception as e:
        print(f"❌ Critical Error during ingestion: {e}")

    print("==================================================")
    print(f"🎉 작업 완료! 총 {total_chunks}개의 청킹 데이터가 RDS에 저장되었습니다.")
    print("==================================================")

if __name__ == "__main__":
    load_and_ingest_data()