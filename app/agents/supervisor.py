from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from app.core.prompts import SUPERVISOR_PROMPT
from app.schemas.state import GraphState
from app.core.config import MODEL_FAST

class IntentRoute(BaseModel):
    """
    구조화된 라우팅 결정을 강제하기 위한 출력 파서(Output Parser) 스키마입니다.
    """
    intent: Literal["ROBOTICS", "WELDING", "ELECTRICAL", "GENERAL"] = Field(
        description="사용자 질의의 도메인 또는 의도입니다."
    )

def create_intent_classifier():
    """
    사용자의 질의를 기반으로 의도를 4가지 카테고리(ROBOTICS, WELDING, ELECTRICAL, GENERAL) 중 하나로 분류하는 의도 분류기(Supervisor)를 생성합니다.
    """
    # 실제 환경에서는 배포된 LLM이나 설정된 객체를 사용합니다.
    llm = ChatOpenAI(model=MODEL_FAST, temperature=0)
    
    # app/core/prompts.py에서 전역 시스템 프롬프트를 불러옵니다.
    prompt = ChatPromptTemplate.from_messages([
        ("system", SUPERVISOR_PROMPT),
        ("human", "{query}")
    ])
    
    # 구조화된 출력 파서 적용 (IntentRoute 모델로 강제 변환시켜 반환)
    classifier_chain = prompt | llm.with_structured_output(IntentRoute)
    
    return classifier_chain

async def route_query(query: str) -> str:
    """
    의도 분류기를 실행하고 라우팅 결정을 반환합니다. (기존 단일 체인 용 API)
    """
    classifier = create_intent_classifier()
    result = await classifier.ainvoke({"query": query})
    return result.intent

async def supervisor_node(state: GraphState) -> dict:
    """
    LangGraph용 Supervisor 노드 (async).
    """
    print("--- [Node: Supervisor] 의도 분류를 시작합니다 ---")
    messages = state.get("messages", [])
    if not messages:
        return {"category": "general"}

    last_message = messages[-1].content
    classifier = create_intent_classifier()
    result = await classifier.ainvoke({"query": last_message})  # [FIX] async

    print(f"[Node: Supervisor] 결정된 라우팅 카테고리: {result.intent.lower()}")
    return {"category": result.intent.lower()}
