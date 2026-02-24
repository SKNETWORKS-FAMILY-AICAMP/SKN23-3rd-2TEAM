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

async def generate_general_answer(query: str) -> str:
    """
    RAG 검색 없이 즉시 일반 안내 멘트를 생성합니다 (async).
    """
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", GENERAL_PROMPT),
        ("human", "{query}")
    ])
    
    chain = prompt | llm
    
    print("[General Agent] 일반 대화 회피 로직을 실행 중입니다...")
    response = await chain.ainvoke({"query": query})  # [FIX] async
    return response.content

async def general_node(state: GraphState) -> dict:
    """
    LangGraph용 일반 대화(General) 우회 노드 (async).
    """
    print("--- [Node: General] RAG 파이프라인 우회 (Bypass) 시작 ---")
    messages = state.get("messages", [])
    query = messages[-1].content if messages else ""

    generated_answer = await generate_general_answer(query)  # [FIX] async

    return {
        "generated_answer": generated_answer,
        "is_hallucinated": False
    }
