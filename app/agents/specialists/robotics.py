# ============================================================
# [로봇 전문가 에이전트] 필수 Import 목록
#   from langchain_core.prompts import ChatPromptTemplate  -- 프롬프트 체인 구성
#   from langchain_openai import ChatOpenAI               -- OpenAI LLM 호출 (gpt-4o)
#   from app.core.prompts import ROBOTICS_SPECIALIST_PROMPT  -- 도메인 전용 프롬프트
#   from app.schemas.state import GraphState              -- LangGraph 상태 타입
#   from app.rag.pipeline import run_rag_pipeline         -- RAG 문서 검색
#   from app.core.security import verify_hallucination    -- 환각 검증 (선택 사용)
# ============================================================
from app.rag.pipeline import run_rag_pipeline
from app.core.security import verify_hallucination
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from app.core.prompts import ROBOTICS_SPECIALIST_PROMPT
from app.schemas.state import GraphState
from app.core.config import get_model_accurate

async def generate_robotics_answer(query: str, context: str, chat_history: str = "") -> str:
    """RAG Context 및 대화 맥락을 사용하여 로봇 특화 답변을 생성합니다 (async)."""
    llm = ChatOpenAI(model=get_model_accurate(), temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", ROBOTICS_SPECIALIST_PROMPT),
        ("human", "{query}")
    ])
    chain = prompt | llm
    print("[Robotics Agent] 로봇 전문가가 답변을 생성 중입니다...")
    response = await chain.ainvoke({
        "context": context, 
        "chat_history": chat_history, 
        "query": query
    })
    return response.content

# ── 도메인 불일치 감지 키워드 ──
# 이 키워드가 context에 포함되면 "관련 문서를 찾지 못한 것"으로 판단
DOMAIN_MISMATCH_SIGNALS = [
    "(관련 매뉴얼 없음)",
    "(관련 문서 없음)",
    "관련 내용을 찾을 수 없",
    "해당 도메인 문서가 없",
]

def detect_domain_mismatch(context: str, domain: str) -> bool:
    """
    검색된 context가 해당 도메인과 전혀 맞지 않는지 감지합니다.
    - context가 비어있거나 불일치 신호 키워드가 포함된 경우 True 반환
    - True 반환 시 graph가 supervisor로 재라우팅하여 재분류 시도
    """
    if not context or len(context.strip()) < 20:
        print(f"[DomainGuard] ⚠️ context 없음 → 도메인 불일치 감지 (domain={domain})")
        return True
    for signal in DOMAIN_MISMATCH_SIGNALS:
        if signal in context:
            print(f"[DomainGuard] ⚠️ 불일치 신호 감지: '{signal}' → domain={domain}")
            return True
    return False

async def robotics_node(state: GraphState) -> dict:
    """
    LangGraph용 로봇 전문가(Robotics) 노드 (async).
    """
    print("--- [Node: Robotics] 로봇 전문가 답변 생성 ---")
    messages = state.get("messages", [])
    original_query = messages[-1].content if messages else ""

    # [원본 질문 보존] 최초 진입 시에만 저장 (멀티턴 피드백 루프 내 의도 유지)
    original_question = state.get("original_question") or original_query

    # [Memory Injection] 이전 대화 기록 확보
    history_msgs = messages[:-1]
    chat_history = "\n".join([
        f"{'사용자' if msg.type == 'human' else 'AI'}: {msg.content}"
        for msg in history_msgs
    ]) if history_msgs else "이전 대화 없음"

    # Query Rewriter가 확장한 쿼리를 우선 사용
    search_query = state.get("rewritten_query") or original_query
    print(f"[Robotics] 검색 쿼리: '{search_query}'")

    # 메타데이터 필터 추출
    filters = {}
    sq_lower = search_query.lower()
    if "hi6" in sq_lower:
        filters = {"model_name": "Hi6"}
    elif "hi5" in sq_lower:
        filters = {"model_name": "Hi5"}

    # RAG 검색
    context, max_score = run_rag_pipeline(search_query, domain="ROBOT", filters=filters)

    # ① 제로히트(Zero-hit) 조기 종료
    # Reranker 엄갑 통과 문서가 0개이면 LLM 호출 없이 즉시 피드백 루프 진입
    if not context or "(관련 매뉴얼 없음)" in context or len(context.strip()) < 30:
        sq = search_query[:60]
        oq = original_question[:40]
        feedback_msg = (
            f"로보틱스 도메인 검색 실패 (결과 0건): {sq!r}. "
            f"원본 질문 {oq!r}의 에러코드/모델명/브랜드명을 더 구체적으로 쿼리를 재작성하세요."
        )
        print("[Robotics] Zero-hit detected: LLM 호출 없이 피드백 루프 진입")
        return {
            "context":           "(검색 결과 없음)",
            "generated_answer":  "",
            "is_hallucinated":   True,
            "retry_count":       state.get("retry_count", 0) + 1,
            "verifier_feedback": feedback_msg,
            "domain_mismatch":   False,
            "original_question": original_question,
            "reranker_score":    max_score,
        }

    # ② 도메인 불일치 감지 — context가 로봇 관련 내용이 아니면 supervisor 재분류
    if detect_domain_mismatch(context, domain="robotics"):
        print("[Robotics] ❌ 도메인 불일치 → supervisor 재분류 요청")
        return {
            "context":          context,
            "generated_answer": "",
            "domain_mismatch":  True,
            "original_question": original_question,
            "reranker_score":   max_score,
        }

    # ② 정상: 도메인 일치 → 답변 생성
    generated_answer = await generate_robotics_answer(original_query, context, chat_history)
    return {
        "context":           context,
        "generated_answer":  generated_answer,
        "domain_mismatch":   False,
        "original_question": original_question,
        "reranker_score":    max_score,
    }
