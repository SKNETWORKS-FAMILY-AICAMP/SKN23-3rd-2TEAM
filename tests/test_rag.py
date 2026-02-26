import os
import sys
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# 프로젝트 루트를 sys.path에 추가하여 app 모듈을 불러올 수 있게 설정
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from app.vectorstore.pgvector_store import PGVectorStoreManager

# 1. 환경 변수 로드
load_dotenv()

def safe_str(s):
    if not isinstance(s, str):
        return str(s)
    return s.encode("utf-8", "ignore").decode("utf-8")

def format_docs(docs):
    cleaned_contents = []
    for doc in docs:
        content = safe_str(doc.page_content)
        h1 = safe_str(doc.metadata.get("Header 1", ""))
        h2 = safe_str(doc.metadata.get("Header 2", ""))
        h3 = safe_str(doc.metadata.get("Header 3", ""))
        source = safe_str(doc.metadata.get("source_file", "알 수 없음"))
        
        header_info = f"[출처: {source} | 섹션: {h1} > {h2} > {h3}]\n"
        cleaned_contents.append(header_info + content)
    return "\n\n---\n\n".join(cleaned_contents)

def get_rag_chain(retriever):
    llm = ChatOpenAI(model="gpt-4o", temperature=0) # gpt-5.1은 존재하지 않으므로 수정
    
    template = """당신은 현대로보틱스 로봇 시스템(Hi5, Hi5a, Hi6) 및 NCS 용접 표준 분야의 최고 기술 전문가 에이전트입니다.
    제공된 [Context]는 당신이 학습한 매뉴얼 데이터입니다. 사용자는 현장의 로봇 운영자 또는 엔지니어입니다.

    [답변 원칙]
    1. **스마트 태그 활용**: [Context] 상단의 태그를 분석하여 질문에 맞는 모델 정보를 제공하세요.
    2. **트러블슈팅 구조화**: [에러 의미/증상] -> [발생 원인] -> [해결 및 조치 방법] 순서로 작성하세요.
    3. **정직한 응답**: 정보가 없다면 "현재 지식 베이스에 해당 내용이 없습니다."라고 답하세요.

    [Context]:
    {context}

    질문: {question}
    
    답변:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    return (
        {
            "context": retriever | format_docs, 
            "question": RunnablePassthrough() | RunnableLambda(safe_str)
        }
        | prompt
        | llm
        | StrOutputParser()
    )

def run_rag_test():
    print("🚀 RDS pgvector 기반 RAG 테스트를 시작합니다...")
    
    try:
        with PGVectorStoreManager(collection_name="welding_robotics_manuals") as vector_store:
            retriever = vector_store.as_retriever(search_kwargs={"k": 5})
            rag_chain = get_rag_chain(retriever)

            if len(sys.argv) > 1:
                query = " ".join(sys.argv[1:])
                print(f"\n질문: {query}")
                response = rag_chain.invoke(query)
                print(f"답변:\n{response}\n")
                return

            print("준비 완료! 질문을 입력하세요. (종료: 'exit')")
            while True:
                query = input("\n질문: ")
                if query.lower() in ["exit", "quit", "종료"]:
                    break
                if not query.strip(): continue

                print("분석 중...", end="", flush=True)
                response = rag_chain.invoke(query)
                print("\r" + " " * 20 + "\r", end="") 
                print(f"답변:\n{response}")
                
                docs = retriever.invoke(query)
                seen_headers = set()
                print("\n[정보 근거 섹션]")
                for doc in docs:
                    source = doc.metadata.get("source_file", "Unknown")
                    h1 = doc.metadata.get("Header 1", "")
                    h2 = doc.metadata.get("Header 2", "")
                    info = f"- {source} ({h1} > {h2})"
                    if info not in seen_headers:
                        print(info)
                        seen_headers.add(info)

    except Exception as e:
        print(f"❌ 에러 발생: {e}")

if __name__ == "__main__":
    run_rag_test()