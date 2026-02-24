import os
from typing import List, Callable, Any
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from app.ingest.chunking import chunk_markdown_document

def ingest_markdown_directory(
    directory_path: str, 
    save_to_vectorstore: Callable[[List[Document], Any], None] = None
) -> List[Document]:
    """
    지정된 디렉토리의 .md 파일들을 스캔하여 AI 처리용 List[Document]로 반환합니다.
    DB 저장이 필요할 경우, save_to_vectorstore 콜백 함수를 주입받아 처리합니다(의존성 주입).
    """
    print(f"--- [Ingest Pipeline] 디렉토리 스캔 시작: {directory_path} ---")
    if not os.path.exists(directory_path):
        print(f"[Ingest Warning] 디렉토리가 존재하지 않습니다: {directory_path}")
        return []
        
    loader = DirectoryLoader(directory_path, glob="**/*.md", loader_cls=TextLoader)
    raw_docs = loader.load()
    print(f"[Ingest Pipeline] 로드된 문서 개수: {len(raw_docs)}개")
    
    all_chunks = []
    
    # 2. 청킹 및 스마트 메타데이터 주입
    for doc in raw_docs:
        source = doc.metadata.get('source', 'Unknown')
        print(f"[Ingest] 파싱 중: {source}")
        chunks = chunk_markdown_document(doc.page_content, source_path=source)
        all_chunks.extend(chunks)
        
    print(f"--- [Ingest Pipeline] 총 {len(all_chunks)}개의 정교한 청크(Chunk)가 생성되었습니다. ---")
    
    # 3. 임베딩 및 벡터 저장 (의존성 주입된 인프라 함수 호출)
    if save_to_vectorstore and all_chunks:
        # OpenAI 임베딩 로드 (실제 인프라 연결시 API KEY 필요)
        try:
            from langchain_openai import OpenAIEmbeddings
            embeddings = OpenAIEmbeddings()
            print(f"--- [Ingest Pipeline] 주입된 Vector DB 저장 함수 호출 ---")
            save_to_vectorstore(all_chunks, embeddings)
        except Exception as e:
            print(f"[Ingest Error] 저장 중 에러가 발생했습니다: {e}")
            
    return all_chunks

# ----------------- Mock 테스트 블록 -----------------
if __name__ == "__main__":
    print("\n================== [Mock Test: Advanced Markdown Chunking] ==================")
    fake_markdown = """# 1장. Hi6 로봇 유지보수
본 장에서는 Hi6 제어기의 기초적인 유지보수에 대해 설명합니다.

## 1.1 배터리 교체 방법
로봇 모터의 엔코더 배터리 전압이 낮을 경우(E012 에러) 다음 순서에 따라 배터리를 교체해야 합니다.

| 단계 | 작업 내용 | 주의 사항 |
|---|---|---|
| 1 | 로봇 전원 차단 | 반드시 Main Power를 OFF 하십시오. |
| 2 | 베이스 커버 분리 | 4개의 볼트를 M5 렌치로 풀어냅니다. |
| 3 | 배터리 커넥터 분리 | 구형 배터리를 새로운 정품 배터리로 교체합니다. |
| 4 | 완료 및 알람 리셋 | 티칭 팬던트에서 E012 알람을 리셋합니다. |

안전에 유의하여 작업하십시오.
"""
    
    print("1. 가상 마크다운 문자열 로드 완료")
    
    # 가상의 GPU 서버 산출물 경로 부여
    fake_path = "domain/robotics/docs/Hi6_Operation_TP630.md"
    
    print(f"2. 청킹 및 메타데이터 자동 주입 실행 (가상 파일: {fake_path})")
    mock_chunks = chunk_markdown_document(fake_markdown, source_path=fake_path)
    
    print(f"\n3. 청킹 결과 분석 ({len(mock_chunks)} chunks):")
    for i, chunk in enumerate(mock_chunks, 1):
        print(f"\n[Chunk {i} Data]")
        print(f" > Domain:        {chunk.metadata.get('domain')}")
        print(f" > Model Name:    {chunk.metadata.get('model_name')}")
        print(f" > Chapter Tree:  {chunk.metadata.get('chapter_path')}")
        print(f" > Full Metadata: {chunk.metadata}")
        print(f" > Page Content:\n{chunk.page_content}")
        print("-" * 50)
    
    print("==========================================================================")
