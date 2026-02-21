import os
import sys
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# 1. 환경 변수 로드
load_dotenv()

def safe_str(s):
    """
    유니코드 인코딩 에러를 방지하기 위해 문자열을 안전하게 정제합니다.
    써로게이트 문자(surrogates) 등 인코딩 불가능한 문자를 제거합니다.
    """
    if not isinstance(s, str):
        return str(s)
    # encode('utf-8', 'ignore')는 에러를 무시하고 인코딩한 뒤 다시 디코딩
    return s.encode("utf-8", "ignore").decode("utf-8")

def format_docs(docs):
    """
    1. 인코딩 에러 방지를 위한 텍스트 정제
    2. 주제(Header) 정보를 본문에 결합하여 LLM의 문맥 이해도 향상
    """
    cleaned_contents = []
    for doc in docs:
        # 본문 및 메타데이터 정제 (모든 필드에 대해 safe_str 적용)
        content = safe_str(doc.page_content)
        
        h1 = safe_str(doc.metadata.get("Header 1", ""))
        h2 = safe_str(doc.metadata.get("Header 2", ""))
        h3 = safe_str(doc.metadata.get("Header 3", ""))
        source = safe_str(doc.metadata.get("source", "알 수 없음"))
        
        # 본문 상단에 문맥 정보 주입
        header_info = f"[출처: {source} | 섹션: {h1} > {h2} > {h3}]\n"
        cleaned_contents.append(header_info + content)
        
    return "\n\n---\n\n".join(cleaned_contents)

def get_rag_chain(retriever):
    """
    RAG(Retrieval-Augmented Generation) 파이프라인(체인)을 생성하는 함수입니다.
    사용자 질문 -> 검색기(Retriever)를 통한 관련 문서 추출 -> 프롬프트 조합 -> LLM 답변 생성의 흐름을 정의합니다.
    """
    # LLM 모델 설정 (gpt-4o-mini 등 존재하는 모델 이름으로 변경하는 것이 좋습니다)
    llm = ChatOpenAI(model="gpt-5.1", temperature=0)
    
    # 챗봇의 페르소나와 답변 원칙을 정의한 시스템 프롬프트 템플릿
    template = """당신은 현대로보틱스 로봇 시스템(Hi5, Hi5a, Hi6) 및 NCS 용접 표준 분야의 최고 기술 전문가 에이전트입니다.
    제공된 [Context]는 당신이 학습한 매뉴얼 데이터이며, 각 문단 앞에는 `[브랜드 모델 영역 | 문서 섹션]` 형태의 출처 태그가 붙어 있습니다.
    사용자는 현장의 로봇 운영자 또는 엔지니어입니다.

    [답변 원칙]
    1. **스마트 태그 활용**: [Context] 상단의 태그(예: `[HD Hi6 Robot | ...]`)를 분석하여, 질문자가 묻는 모델(Hi5a, Hi6 등)에 정확히 일치하는 답변을 제공하세요. 두 모델의 내용이 다르다면 비교해서 설명하세요.
    2. **에이전트 페르소나**: "어떤 로봇을 사용할 줄 알아?", "너의 역할이 뭐야?" 같은 질문에는 [Context]의 태그 정보들을 참고하여 "저는 현대로보틱스 Hi5, Hi6 제어기 매뉴얼과 NCS 로봇 용접 표준 지식을 보유하고 있습니다"라고 자연스럽게 답변하세요.
    3. **트러블슈팅 구조화**: 에러 코드나 기계 고장 관련 질문은 반드시 [에러 의미/증상] -> [발생 원인] -> [해결 및 조치 방법] 순서로 가독성 있게 작성하세요.
    4. **수치 및 표 데이터**: 전류, 전압, 파라미터 번호 등은 문서 그대로 정확하게 인용하고, 복잡한 정보는 마크다운 표(|---|)로 정리하세요.
    5. **출처 인용**: 답변이 끝나면 참조한 모델명과 문서의 섹션 정보를 간략히 명시하세요.
    6. **정직한 응답**: 기술적인 질문에 대해 [Context]에 정말로 단서가 없다면 절대 추측하지 말고 "현재 지식 베이스에 해당 내용이 없습니다."라고 답하세요.

    [Context]:
    {context}

    질문: {question}
    
    답변:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # LangChain의 LCEL(LangChain Expression Language) 문법을 사용해 파이프라인 구성
    return (
        {
            # 컨텍스트: 검색기(retriever)가 찾은 문서를 format_docs 함수를 거쳐 문자열로 변환
            "context": retriever | format_docs, 
            # 질문: 사용자의 질문을 safe_str 함수로 안전하게 정제 (인코딩 에러 방지)
            "question": RunnablePassthrough() | RunnableLambda(safe_str)
        }
        | prompt             # 변수들이 주입된 프롬프트 완성
        | llm                # LLM 모델을 통해 답변 텍스트 생성
        | StrOutputParser()  # 생성된 응답 객체에서 최종 문자열만 추출
    )

def run_rag_test():
    """
    RAG 파이프라인을 터미널에서 테스트할 수 있도록 하는 실행 함수입니다.
    CLI로 매개변수를 넘기거나, 대화형 모드로 계속해서 질문을 할 수 있습니다.
    """
    db_dir = "./db_welding"
    if not os.path.exists(db_dir):
        print(f"데이터베이스 폴더({db_dir})가 없습니다. vector_store.py를 먼저 실행하세요.")
        return

    print("용접 지식 베이스(Chroma)를 로드 중입니다...")
    # 저장된 DB를 불러와 초기화
    vector_db = Chroma(
        persist_directory=db_dir,
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-small")
    )

    # 검색기(Retriever) 생성 및 RAG 체인 연결 (가장 유사한 5개 문서 검색)
    retriever = vector_db.as_retriever(search_kwargs={"k": 5})
    rag_chain = get_rag_chain(retriever)

    # 1. 터미널 인자로 질문을 받은 경우 (예: python test_rag.py "용접 전류란?") - 단발성 질의 (CLI 모드)
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print(f"\n질문: {query}")
        print("분석 중...", flush=True)
        response = rag_chain.invoke(query)
        print(f"답변:\n{response}\n")
        return

    # 2. 대화형 모드 (터미널에서 exit를 칠 때까지 무한 반복하며 질의 가능)
    print("준비 완료! 질문을 입력하세요. (종료: 'exit')")
    while True:
        query = input("\n질문: ")
        # 종료 커맨드 인식
        if query.lower() in ["exit", "quit", "종료"]:
            break
        # 빈 입력 무시
        if not query.strip(): continue

        print("분석 중...", end="", flush=True)
        # RAG 체인 실행하여 답변 구하기
        response = rag_chain.invoke(query)
        # 진행 중 메시지 덮어쓰기 지우기
        print("\r" + " " * 20 + "\r", end="") 
        print(f"답변:\n{response}")
        
        # 출처 표시: LLM이 참고한 정보의 근거(섹션 헤더 등)를 요약하여 보여줍니다
        docs = retriever.invoke(query)
        seen_headers = set()
        print("\n[정보 근거 섹션]")
        for doc in docs:
            source = doc.metadata.get("source", "Unknown")
            h1 = doc.metadata.get("Header 1", "")
            h2 = doc.metadata.get("Header 2", "")
            info = f"- {source} ({h1} > {h2})"
            if info not in seen_headers:
                print(info)
                seen_headers.add(info)

if __name__ == "__main__":
    run_rag_test()