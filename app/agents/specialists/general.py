# ============================================================
# [일반(General) 우회 에이전트] 필수 Import 목록
#   from langchain_core.prompts import ChatPromptTemplate  -- 프롬프트 체인 구성
#   from langchain_openai import ChatOpenAI               -- OpenAI LLM 호출 (gpt-4o)
#   from app.core.prompts import GENERAL_PROMPT           -- 일반 안내원 프롬프트
#   from app.schemas.state import GraphState              -- LangGraph 상태 타입
#   [RAG 및 Verifier 미사용: 이 에이전트는 DB 검색을 완전히 우회(Bypass)합니다]
# ============================================================
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from app.core.prompts import GENERAL_PROMPT
from app.schemas.state import GraphState
from app.core.config import MODEL_FAST

from langchain_community.tools.tavily_search import TavilySearchResults
from app.core.config import MODEL_FAST, MODEL_ACCURATE, TAVILY_API_KEY

async def generate_general_answer(query: str, chat_history: str = "") -> str:
    """
    [수정됨] 대화 맥락을 활용하며, 검색 없이 LLM 단독 답변으로 처리합니다.
    """
    # 1. Tavily 검색 도구 초기화 (주석 처리)
    # search = TavilySearchResults(k=3)
    
    try:
        # 2. 검색 수행 및 컨텍스트 생성 (주석 처리)
        # search_results = await search.ainvoke({"query": query})
        # context = "\n".join([f"Source: {r['url']}\nContent: {r['content']}" for r in search_results])
        
        # 3. LLM을 통한 답변 생성 (검색 결과 없이 직접 답변)
        llm = ChatOpenAI(model=MODEL_ACCURATE, temperature=0.7)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", GENERAL_PROMPT),
            ("human", "대화 명세:\n{chat_history}\n\n[현재 질문]\n{query}")
        ])
        
        chain = prompt | llm
        response = await chain.ainvoke({"query": query, "chat_history": chat_history})
        return response.content
        
    except Exception as e:
        print(f"[General Agent] Error: {e} -> Fallback to basic LLM")
        llm = ChatOpenAI(model=MODEL_FAST, temperature=0.7)
        prompt = ChatPromptTemplate.from_messages([
            ("system", GENERAL_PROMPT),
            ("human", "대화 명세:\n{chat_history}\n\n[현재 질문]\n{query}")
        ])
        chain = prompt | llm
        response = await chain.ainvoke({"query": query, "chat_history": chat_history})
        return response.content

async def general_node(state: GraphState) -> dict:
    """
    LangGraph용 일반 대화(General) 우회 노드 (async).
    """
    print("--- [Node: General] RAG 파이프라인 우회 (Bypass) 시작 ---")
    messages = state.get("messages", [])
    query = messages[-1].content if messages else ""

    # [Memory Injection] 이전 대화 기록 확보
    history_msgs = messages[:-1]
    chat_history = "\n".join([
        f"{'사용자' if msg.type == 'human' else 'AI'}: {msg.content}"
        for msg in history_msgs
    ]) if history_msgs else "이전 대화 없음"

    generated_answer = await generate_general_answer(query, chat_history)

    return {
        "generated_answer": generated_answer,
        "is_hallucinated": False
    }
