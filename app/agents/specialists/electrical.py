# ============================================================
# [전기 전문가 에이전트] 필수 Import 목록
#   from langchain_core.prompts import ChatPromptTemplate   -- 프롬프트 체인 구성
#   from langchain_openai import ChatOpenAI                -- OpenAI LLM 호출 (gpt-4o)
#   from app.core.prompts import ELECTRICAL_SPECIALIST_PROMPT -- 도메인 전용 프롬프트
#   from app.schemas.state import GraphState               -- LangGraph 상태 타입
#   from app.rag.pipeline import run_rag_pipeline          -- RAG 문서 검색
#   from app.core.security import verify_hallucination     -- 환각 검증 (선택 사용)
# ============================================================
import asyncio
from app.rag.pipeline import run_rag_pipeline
from app.core.security import verify_hallucination
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from app.core.prompts import ELECTRICAL_SPECIALIST_PROMPT
from app.schemas.state import GraphState
from app.core.config import get_model_accurate

async def generate_electrical_answer(query: str, context: str, chat_history: str = "") -> str:
    """RAG Context 및 대화 맥락을 사용하여 전기/보전 특화 답변을 생성합니다 (async)."""
    llm = ChatOpenAI(model=get_model_accurate(), temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", ELECTRICAL_SPECIALIST_PROMPT),
        ("human", "대화 명세:\n{chat_history}\n\n[현재 질문]\n{query}")
    ])
    chain = prompt | llm
    print("[Electrical Agent] 전기/전장 전문가가 답변을 생성 중입니다...")
    response = await chain.ainvoke({"context": context, "query": query, "chat_history": chat_history})
    return response.content


async def electrical_node(state: GraphState) -> dict:
    """
    LangGraph용 전기/전장 전문가(Electrical) 노드 (async).
    """
    print("--- [Node: Electrical] 전기/전장 전문가 답변 생성 ---")
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
    print(f"[Electrical] 검색 쿼리: '{search_query}'")

    context, max_score = await asyncio.to_thread(run_rag_pipeline, search_query, domain="ELECTRICAL")

    # 제로히트(Zero-hit) 조기 종료
    if not context or len(context.strip()) < 30:
        sq = search_query[:50]
        oq = original_question[:40]
        feedback_msg = (
            f"전기 도메인 검색 실패 (결과 0건): {sq!r}. "
            f"원본 질문 {oq!r}의 제어반 형식/센서 종류/PLC I/O 번호 등 "
            "구체적 키워드를 포함하여 쿼리를 재작성하세요."
        )
        print("[Electrical] Zero-hit: 피드백 루프 진입")
        return {
            "context": "(검색 결과 없음)", "generated_answer": "",
            "is_hallucinated": True, "retry_count": state.get("retry_count",0)+1,
            "verifier_feedback": feedback_msg, "domain_mismatch": False,
            "original_question": original_question,
            "reranker_score": max_score,
        }

    MISMATCH = ["(관련 매뉴얼 없음)","(관련 문서 없음)","관련 내용을 찾을 수 없"]
    if not context or any(s in context for s in MISMATCH):
        print("[Electrical] ❌ 도메인 불일치 → supervisor 재분류 요청")
        return {
            "context": context, "generated_answer": "",
            "domain_mismatch": True, "original_question": original_question,
            "reranker_score": max_score,
        }

    generated_answer = await generate_electrical_answer(original_query, context, chat_history)
    return {
        "context": context, "generated_answer": generated_answer,
        "domain_mismatch": False, "original_question": original_question,
        "reranker_score": max_score,
    }
