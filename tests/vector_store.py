import os
import shutil
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

load_dotenv()

def build_standardized_db():
    """
    파싱된 마크다운 문서들(data/2_llama_md)을 읽어들여 
    Chroma DB(db_welding)라는 벡터 스토어(Vector Store)로 구축하는 과정을 담당하는 함수입니다.
    """
    input_dir = "data/2_llama_md" # 마크다운 파일이 있는 입력 디렉토리
    db_dir = "./db_welding"       # 벡터 데이터베이스가 저장될 경로

    # 1. 초기화 로직: 이전 DB가 남아있을 경우 덮어쓰기 문제 등을 방지하기 위해 완전히 삭제합니다.
    if os.path.exists(db_dir):
        shutil.rmtree(db_dir)

    # 입력 폴더가 존재하는지 혹시 모를 상황 대비 검사
    if not os.path.exists(input_dir):
        print(f"입력 폴더({input_dir})가 없습니다.")
        return

    # 2. 하위 폴더를 포함하여 모든 마크다운 파일 탐색 (os.walk 적용)
    md_files_info = []
    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            if filename.endswith(".md"): # 확장자가 .md로 끝나는 파일만 수집
                md_files_info.append({
                    "root": root,
                    "filename": filename
                })

    all_docs = [] # 최종적으로 DB에 저장될 도큐먼트 리스트

    for file_info in md_files_info:
        root = file_info["root"]
        filename = file_info["filename"]
        file_path = os.path.join(root, filename)

        # 3. 폴더 경로를 기반으로 한 지능형 메타데이터(Metadata) 추출
        # 폴더 구조에 정보가 담겨있는 것을 활용하여, 나중에 검색(Retriever) 결과의 품질을 높입니다.
        # 예: root가 "data/2_llama_md/robot/Hi6" 일 경우
        rel_path = os.path.relpath(root, input_dir) # 기준 폴더를 제외한 경로 (예: "robot/Hi6")
        path_parts = rel_path.split(os.sep)         # ["robot", "Hi6"]

        # NCS 파일인지 판단 (파일명에 LM이 들어가면 일반적인 NCS 규격 문서로 분류)
        if "LM16" in filename:
            brand = "NCS"
            model = "General"
            category = "Welding"
        else:
            brand = "HD" # 현대로보틱스
            # 폴더명 패턴에서 모델명 추출 (예: Hi6, Hi5a, HH_Series)
            model = path_parts[-1] if len(path_parts) > 0 and path_parts[0] != "." else "General"
            # 첫번째 하위 폴더를 카테고리로 지정
            category = path_parts[0] if len(path_parts) > 1 else "Robot"

        # 청크(Chunk)에 붙여둘 태그 정보 정리
        metadata_tags = {
            "brand": brand,
            "model": model,
            "category": category,
            "source": filename # 출처를 표시하기 위함
        }

        # 4. 파일 읽기
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        # 5. 마크다운 계층 분할 (1차 분할)
        # MarkdownHeaderTextSplitter는 #, ##, ### 등을 기준으로 문장을 청크(Chunk)로 나눕니다.
        # 이를 통해 한 헤더 내의 문맥이 깨지지 않게 보존하는 역할을 합니다.
        md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")])
        header_splits = md_splitter.split_text(text)

        # 6. 정밀 분할 및 태깅 (2차 분할)
        # 위에서 헤더로 나누었지만, 섹션 내의 내용이 너무 길어 LLM이 처리하기 무거울 수 있습니다.
        # 따라서 글자수 기준으로 다시 자릅니다 (최대 1000자, 중복 150자 유지하여 문맥 단절 최소화)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        final_splits = text_splitter.split_documents(header_splits)

        for split in final_splits:
            split.metadata.update(metadata_tags) # 위에서 파악한 지능형 태그(브랜드, 모델명 등) 데이터를 주입
            
            # 앞선 1차 분할에서 얻은 헤더 정보들을 읽어옵니다.
            h1 = split.metadata.get("Header 1", "")
            h2 = split.metadata.get("Header 2", "")
            h3 = split.metadata.get("Header 3", "")
            
            # [HD Hi6 Robot | 특정 헤더 구조] 형태로 본문(Content)의 제일 앞부분에 문맥을 강제로 주입합니다.
            # 이 처리 덕분에 LLM은 이 자투리 문장(청크)이 어떤 문서의 어느 파트에서 왔는지 알 수 있습니다.
            context_prefix = f"[{brand} {model} {category} | {h1} > {h2} > {h3}]\n"
            split.page_content = context_prefix + split.page_content

        all_docs.extend(final_splits) # 완성된 문서 청크들을 최종 리스트에 누적

    # 7. 생성된 모든 청크(도큐먼트)를 임베딩(Embedding) 모델을 거쳐 Chroma DB에 저장
    Chroma.from_documents(
        documents=all_docs,
        embedding=OpenAIEmbeddings(model="text-embedding-3-small"), # 고성능 최신 텍스트 임베딩 모델
        persist_directory=db_dir # 폴더 형태로 로컬 저장
    )


if __name__ == "__main__":
    build_standardized_db()