# ============================================================
# [용접 전문가 에이전트] 필수 Import 목록
#   from langchain_core.prompts import ChatPromptTemplate  -- 프롬프트 체인 구성
#   from langchain_openai import ChatOpenAI               -- OpenAI LLM 호출 (gpt-4o)
#   from app.core.prompts import WELDING_SPECIALIST_PROMPT  -- 도메인 전용 프롬프트
#   from app.schemas.state import GraphState              -- LangGraph 상태 타입
#   from app.rag.pipeline import run_rag_pipeline         -- RAG 문서 검색
#   from app.core.security import verify_hallucination    -- 환각 검증 (선택 사용)
# ============================================================
from app.rag.pipeline import run_rag_pipeline
from app.core.security import verify_hallucination
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from app.core.prompts import WELDING_SPECIALIST_PROMPT
from app.schemas.state import GraphState
from app.core.config import get_model_accurate

async def generate_welding_answer(query: str, context: str, chat_history: str = "") -> str:
    """RAG Context 및 대화 맥락을 사용하여 용접 특화 답변을 생성합니다 (async)."""
    llm = ChatOpenAI(model=get_model_accurate(), temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", WELDING_SPECIALIST_PROMPT),
        ("human", "대화 명세:\n{chat_history}\n\n[현재 질문]\n{query}")
    ])
    chain = prompt | llm
    print("[Welding Agent] 용접 전문가가 답변을 생성 중입니다...")
    response = await chain.ainvoke({"context": context, "query": query, "chat_history": chat_history})
    return response.content


async def welding_node(state: GraphState) -> dict:
    """
    LangGraph용 용접 전문가(Welding) 노드 (async).
    """
    print("--- [Node: Welding] 용접 전문가 답변 생성 ---")
    messages = state.get("messages", [])
    original_query = messages[-1].content if messages else ""
    original_question = state.get("original_question") or original_query

    # [Memory Injection] 이전 대화 기록 확보
    history_msgs = messages[:-1]
    chat_history = "\n".join([
        f"{'사용자' if msg.type == 'human' else 'AI'}: {msg.content}"
        for msg in history_msgs
    ]) if history_msgs else "이전 대화 없음"

    search_query = state.get("rewritten_query") or original_query
    print(f"[Welding] 검색 쿼리: '{search_query}'")

    context, max_score = run_rag_pipeline(search_query, domain="WELDING")

    # 제로히트(Zero-hit) 조기 종료
    if not context or len(context.strip()) < 30:
        feedback_msg = (
            f"용접(WELDING) 도메인 검색 실패: '\"{search_query[:50]}\"'"
            " 용접 결함명/소모품명/모재 종류를 새 키워드로 용어를 더 구체적으로 포함하여"
            f" '\"{original_question[:40]}\"'의 쿼리를 재작성하세요."
        )
        print(f"[Welding] ❌ Zero-hit → 피드백 루프 진입")
        return {
            "context": "(검색 결과 없음)", "generated_answer": "",
            "is_hallucinated": True, "retry_count": state.get("retry_count",0)+1,
            "verifier_feedback": feedback_msg, "domain_mismatch": False,
            "original_question": original_question,
            "reranker_score": max_score,
        }

    # 도메인 불일치 감지
    MISMATCH = ["(관련 매뉴얼 없음)","(관련 문서 없음)","관련 내용을 찾을 수 없"]
    if not context or any(s in context for s in MISMATCH):
        print("[Welding] ❌ 도메인 불일치 → supervisor 재분류 요청")
        return {
            "context": context, "generated_answer": "",
            "domain_mismatch": True, "original_question": original_question,
            "reranker_score": max_score,
        }

    generated_answer = await generate_welding_answer(original_query, context, chat_history)
    return {
        "context": context, "generated_answer": generated_answer,
        "domain_mismatch": False, "original_question": original_question,
        "reranker_score": max_score,
    }
