import re
import os
from typing import List
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def extract_metadata_from_path(file_path: str) -> dict:
    """
    파일 경로와 이름에서 RAG 필터링용 메타데이터를 추출합니다.
    - domain: 경로 기반 식별 (robotics, welding, electrical)
    - model_name: 파일명에서 정규식으로 고유 모델 추출 (예: 'Hi6_Operation_TP630.md' -> 'Hi6_TP630')
    """
    metadata = {}
    
    # 1. 도메인 추출 (예: 'domain/robotics/docs/...' -> 'robotics')
    path_parts = file_path.split(os.sep)
    domain = "general"
    if "robotics" in path_parts:
        domain = "robotics"
    elif "welding" in path_parts:
        domain = "welding"
    elif "electrical" in path_parts:
        domain = "electrical"
    metadata["domain"] = domain
    
    # 2. 모델명 추출 (정규식 활용)
    # 예: Hi6_Operation_TP630.md -> Hi6_TP630 패턴 추출. 
    # [영문숫자]_[무시]_[영문숫자].md 형태
    filename = os.path.basename(file_path)
    match = re.search(r"([a-zA-Z0-9]+)_.+_([a-zA-Z0-9]+)\.md", filename)
    if match:
        metadata["model_name"] = f"{match.group(1)}_{match.group(2)}"
    else:
        # 패턴에 맞지 않으면 기본 확장자만 제거하여 담기
        metadata["model_name"] = filename.replace(".md", "")
        
    return metadata

def chunk_markdown_document(content: str, source_path: str = "unknown.md") -> List[Document]:
    """
    마크다운 포맷의 문서를 1, 2차에 나누어 표와 목차가 훼손되지 않게 정교하게 청킹합니다.
    """
    # [1차 분할] 마크다운 헤더(h1, h2, h3) 기반 스플리터 적용
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, 
        strip_headers=False # 헤더 텍스트 보존
    )
    md_header_splits = markdown_splitter.split_text(content)
    
    # [메타데이터 주입] 경로 분석 데이터 및 챕터 계층 구조 병합
    base_metadata = extract_metadata_from_path(source_path)
    base_metadata["source"] = source_path
    
    for doc in md_header_splits:
        # 기존 헤더 메타데이터를 'chapter_path' 문자열 하나로 묶기
        chapter_parts = []
        if "Header 1" in doc.metadata:
            chapter_parts.append(doc.metadata["Header 1"])
        if "Header 2" in doc.metadata:
            chapter_parts.append(doc.metadata["Header 2"])
        if "Header 3" in doc.metadata:
            chapter_parts.append(doc.metadata["Header 3"])
            
        chapter_path = " > ".join(chapter_parts) if chapter_parts else "전체"
        
        # 문서 조각에 스마트 메타데이터 덮어쓰기
        new_meta = {**base_metadata, "chapter_path": chapter_path}
        doc.metadata = new_meta

    # [2차 분할] 재귀적 글자수 기반 청킹 적용
    # 마크다운 표(|...|)나 코드 블록의 형태를 살리기 위해 문단과 표 바운더리를 지키는 구분자 순서를 설정합니다.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""]
    )
    
    final_splits = text_splitter.split_documents(md_header_splits)
    
    return final_splits
