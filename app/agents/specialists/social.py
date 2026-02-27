# ============================================================
# [소셜/인사(Social) 전용 에이전트]
# backend/app/agents/specialists/social.py
# ============================================================
# 역할: 인삿말, 감사 인사 등 비기술적 질의에 즉각 응답 (No RAG, No WebSearch)
# ============================================================
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from app.schemas.state import GraphState
from app.core.config import MODEL_FAST

SOCIAL_PROMPT = """당신은 산업용 로봇 6대 브랜드(현대, 야스카와, 두산, ABB, UR, 레인보우) 및 용접/전기 기술지원을 위한 전문 챗봇입니다.
기술적인 질문이 아닌 일반적인 인사, 자기소개, 감사의 말에 대해 따뜻하고 전문적인 어조로 답변하세요.
지원 브랜드와 기술 분야를 언급하며 친절하게 자기소개를 하십시오.

기술적인 도움이 필요할 때는 언제든 구체적인 에러코드나 증상을 말씀해달라고 안내하세요.
답변은 1~2문장 내외로 짧고 간결하게 하세요."""

async def social_node(state: GraphState) -> dict:
    """
    단순 인사 및 일상적 대화에 즉각 응답하는 노드 (async).
    RAG 파이프라인과 웹 검색을 완전히 우회하여 응답 속도를 극대화합니다.
    """
    print("--- [Node: Social] Fast Track 발동 (Sub-second Response) ---")
    messages = state.get("messages", [])
    query = messages[-1].content if messages else "안녕하세요"

    # [Memory Injection] 이전 대화 기록 확보
    history_msgs = messages[:-1]
    chat_history = "\n".join([
        f"{'사용자' if msg.type == 'human' else 'AI'}: {msg.content}"
        for msg in history_msgs
    ]) if history_msgs else "이전 대화 없음"

    llm = ChatOpenAI(model=MODEL_FAST, temperature=0.7, streaming=False)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SOCIAL_PROMPT),
        ("human", "대화 명세:\n{chat_history}\n\n[현재 질문]\n{query}")
    ])
    
    chain = prompt | llm
    response = await chain.ainvoke({"query": query, "chat_history": chat_history})
    
    return {
        "generated_answer": response.content,
        "is_hallucinated": False
    }
