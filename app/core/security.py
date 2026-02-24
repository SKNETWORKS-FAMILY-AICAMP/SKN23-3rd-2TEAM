# ============================================================
# [환각 검증기 — Verifier Feedback 강화판] app/core/security.py
# ============================================================
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from app.core.prompts import HALLUCINATION_VERIFIER_PROMPT
from app.schemas.state import GraphState


class HallucinationEval(BaseModel):
    """환각 검증기(Verifier)의 출력 스키마."""
    is_hallucinated: bool = Field(
        description="답변이 Context에 없는 정보를 포함하거나 모순되면 True, 충실하면 False"
    )
    failure_analysis: str = Field(
        default="",
        description=(
            "is_hallucinated=True일 때만 작성. 검증 실패의 구체적 원인 분석:\n"
            "① 어떤 정보가 Context에 없었는가?\n"
            "② 어떤 키워드/모델명/에러코드를 추가로 검색해야 하는가?\n"
            "③ 검색 쿼리를 어떻게 바꿔야 올바른 문서가 나오는가?\n"
            "한국어 3~5문장으로 작성. 통과(False)이면 빈 문자열 반환."
        )
    )


# ── [FIX] system + human 메시지 분리 → query/context/answer 변수 LLM에 올바르게 전달 ──
# 기존: system 프롬프트에만 넣어 {query}/{context}/{answer} 변수가 치환되지 않는 버그
VERIFIER_PROMPT_ENHANCED = (
    HALLUCINATION_VERIFIER_PROMPT
    + """

[추가 지시 — 실패 시 failure_analysis 작성]
만약 is_hallucinated=True로 판정했다면, 다음 3가지를 포함한 분석을 작성하세요:
① 어떤 구체적 정보(에러코드, 모델명, 부품명, 수치 등)가 Context에 없거나 부족했는가?
② 다음 검색 시 어떤 추가 키워드, 모델명, 혹은 에러코드를 포함해야 하는가?
③ 현재 rewritten_query의 어떤 부분을 바꾸거나 추가하면 더 정확한 문서가 검색될 것인가?
3~5문장, 한국어로 작성하세요.
"""
)

# [FIX] human 메시지에 실제 검증 대상(query/context/answer)을 변수로 전달
VERIFIER_HUMAN_TEMPLATE = """[사용자 원본 질문]
{query}

[RAG 검색 Context]
{context}

[챗봇이 생성한 답변]
{answer}

[참고 — 사용된 검색 쿼리]
{rewritten_query}

위 내용을 바탕으로 답변의 환각 여부를 판단하세요."""


async def verify_hallucination_with_feedback(
    query: str,
    context: str,
    answer: str,
    rewritten_query: str = "",
) -> tuple[bool, str]:
    """
    강화된 환각 검증 함수 (async).

    Returns:
        (is_hallucinated: bool, feedback: str)
    """
    print("[Verifier] 환각 검증 + 실패 원인 분석 중...")

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", VERIFIER_PROMPT_ENHANCED),
        ("human", VERIFIER_HUMAN_TEMPLATE),   # [FIX] 변수 슬롯을 human 메시지로 이동
    ])
    chain = prompt | llm.with_structured_output(HallucinationEval)

    result = await chain.ainvoke({          # [FIX] async 호출
        "query":           query,
        "context":         context,
        "answer":          answer,
        "rewritten_query": rewritten_query,
    })

    if result.is_hallucinated:
        print(f"[Verifier] 실패 — 환각 감지")
        print(f"[Verifier] 분석: {result.failure_analysis[:120]}...")
        return True, result.failure_analysis.strip()
    else:
        print("[Verifier] 통과 — Context에 충실한 답변")
        return False, ""


# 기존 sync 함수 — 단독 테스트용으로 유지 (LangGraph 노드에서는 사용 안 함)
def verify_hallucination(query: str, context: str, answer: str) -> bool:
    """단독 테스트/검증용 동기 함수. LangGraph 노드는 verifier_node를 사용하세요."""
    from langchain_core.prompts import ChatPromptTemplate
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", VERIFIER_PROMPT_ENHANCED),
        ("human", VERIFIER_HUMAN_TEMPLATE),
    ])
    chain = prompt | llm.with_structured_output(HallucinationEval)
    result = chain.invoke({
        "query": query, "context": context,
        "answer": answer, "rewritten_query": "",
    })
    return not result.is_hallucinated


async def verifier_node(state: GraphState) -> dict:
    """
    LangGraph용 환각 검증 노드 (async).

    [출력 상태]
        is_hallucinated:   True/False
        retry_count:       실패 시 +1
        verifier_feedback: 실패 시 분석 문자열, 통과 시 빈 문자열
    """
    print("--- [Node: Verifier] 환각 검증 시작 ---")
    messages        = state.get("messages", [])
    query           = messages[-1].content if messages else ""
    context         = state.get("context", "")
    answer          = state.get("generated_answer", "")
    rewritten_query = state.get("rewritten_query", "")
    retry_count     = state.get("retry_count", 0)

    # Context나 Answer가 없으면 즉시 실패 (Zero-hit 경로와 중복이지만 방어적 처리)
    if not answer or not context or context == "(검색 결과 없음)":
        feedback = (
            "검색 결과(Context)가 비어있습니다. "
            "현재 rewritten_query로는 관련 문서가 검색되지 않았습니다. "
            "제조사명, 에러코드, 모델명을 더 구체적으로 포함하여 쿼리를 재작성하세요. "
            f"[현재 쿼리: '{rewritten_query}']"
        )
        print("[Verifier] Context 없음 — 피드백 생성")
        return {
            "is_hallucinated":   True,
            "retry_count":       retry_count + 1,
            "verifier_feedback": feedback,
        }

    is_hallucinated, feedback = await verify_hallucination_with_feedback(
        query, context, answer, rewritten_query
    )

    new_retry = retry_count + 1 if is_hallucinated else retry_count

    return {
        "is_hallucinated":   is_hallucinated,
        "retry_count":       new_retry,
        "verifier_feedback": feedback,  # 통과 시 ""
    }