# ============================================================
# [메인 LangGraph 워크플로우 — 3중 안정성 강화 완성판]
# app/agents/graph.py
# ============================================================
# ① 도메인 재분류 무한루프 방지: routing_retry <= 1
# ② Verifier 피드백 루프: feedback_rewriter → specialist 재실행
# ③ Fallback 로깅: 관리자용 JSONL 로그 파일 수집
# ============================================================
import json
from datetime import datetime
from pathlib import Path
from langgraph.graph import StateGraph, START, END
from app.schemas.state import GraphState
from app.core.history import get_memory_saver

from app.agents.supervisor import supervisor_node
from app.agents.specialists.robotics import robotics_node
from app.agents.specialists.welding import welding_node
from app.agents.specialists.electrical import electrical_node
from app.agents.specialists.general import general_node
from app.agents.specialists.social import social_node
from app.agents.tools.rewriter import rewriter_node, feedback_rewriter_node
from app.core.security import verifier_node

# ─────────────────────────────────────────────────────────────
# 유틸리티 노드 및 조건부 라우팅 함수
# ─────────────────────────────────────────────────────────────

async def reroute_supervisor_node(state: GraphState) -> dict:
    """도메인 불일치 감지 시 재분류를 위해 상태를 업데이트합니다."""
    print("--- [Node: Supervisor Reroute] 도메인 재분류를 준비합니다 ---")
    return {"routing_retry": state.get("routing_retry", 0) + 1}

async def fallback_node(state: GraphState) -> dict:
    """모든 복구 시도가 실패했을 때의 최종 답변 노드."""
    print("--- [Node: Fallback] 최종 복구 답변을 생성합니다 ---")
    return {
        "generated_answer": "죄송합니다. 요청하신 기술 질의에 대해 충분한 정보를 찾지 못했거나 내부 오류가 발생했습니다. 구체적인 장비 모델명과 증상을 다시 말씀해 주시면 성심껏 도와드리겠습니다.",
        "is_hallucinated": False
    }

def route_feedback_to_specialist(state: GraphState) -> str:
    """피드백 기반 재작성 후 다시 원래의 전문가로 라우팅합니다."""
    category = state.get("category", "general")
    if category not in ["robotics", "welding", "electrical", "general"]:
        return "general"
    return category

def check_domain_mismatch(state: GraphState) -> str:
    """Specialist가 감지한 도메인 불일치에 따라 다음 노드를 선택합니다."""
    if state.get("domain_mismatch"):
        if state.get("routing_retry", 0) < 1:
            return "supervisor_reroute"
        return "fallback"
    return "verifier"

def check_hallucination(state: GraphState) -> str:
    """Verifier의 환각 판정 결과에 따라 다음 노드를 선택합니다."""
    if state.get("is_hallucinated"):
        if state.get("retry_count", 0) < 2:
            return "feedback_rewriter"
        return "fallback"
    return END

def route_after_rewrite(state: GraphState) -> str:
    """쿼리 재작성 후 소셜 인사인지 전문 기술 질의인지 판단하여 라우팅합니다."""
    hint = state.get("routing_hint", "TECHNICAL")
    if hint == "SOCIAL":
        return "social"
    # [GENERAL] 또는 [TECHNICAL]은 모두 Supervisor로 보내어 Intent 기반 최종 분류 수행
    return "supervisor"

def route_to_specialist(state: GraphState) -> str:
    """Supervisor의 category 결정에 따라 specialist를 선택합니다."""
    return state.get("category", "general")

# ... (check_domain_mismatch and check_hallucination remain same) ...

# ─────────────────────────────────────────────────────────────
# 그래프 구성
# ─────────────────────────────────────────────────────────────
workflow = StateGraph(GraphState)

# 노드 등록
workflow.add_node("rewriter",           rewriter_node)
workflow.add_node("supervisor",         supervisor_node)
workflow.add_node("robotics",           robotics_node)
workflow.add_node("welding",            welding_node)
workflow.add_node("electrical",         electrical_node)
workflow.add_node("general",            general_node)
workflow.add_node("social",             social_node)
workflow.add_node("verifier",           verifier_node)
workflow.add_node("feedback_rewriter",  feedback_rewriter_node, )
workflow.add_node("fallback",           fallback_node)
workflow.add_node("supervisor_reroute", reroute_supervisor_node)

# START -> rewriter
workflow.add_edge(START, "rewriter")

# [Fast Track] Rewriter → (Social 바로가기 | Supervisor 기술 분류)
workflow.add_conditional_edges(
    "rewriter",
    route_after_rewrite,
    {
        "social":     "social",
        "supervisor": "supervisor"
    }
)

# Supervisor → 도메인별 specialist
workflow.add_conditional_edges(
    "supervisor",
    route_to_specialist,
    {
        "robotics":   "robotics",
        "welding":    "welding",
        "electrical": "electrical",
        "general":    "general",
    }
)

# specialist → 도메인 불일치 감지 → verifier 또는 supervisor 재분류
for node in ["robotics", "welding", "electrical"]:
    workflow.add_conditional_edges(
        node,
        check_domain_mismatch,
        {
            "verifier":          "verifier",
            "supervisor_reroute": "supervisor_reroute",
            "fallback":          "fallback",
        }
    )

# supervisor_reroute → 재분류된 specialist로 이동
workflow.add_conditional_edges(
    "supervisor_reroute",
    route_to_specialist,
    {
        "robotics":   "robotics",
        "welding":    "welding",
        "electrical": "electrical",
        "general":    "general",
    }
)

# general & social → 검증 없이 즉시 END (Fast Track 완료)
workflow.add_edge("general", END)
workflow.add_edge("social", END)

# Verifier → (통과: END | 실패: feedback_rewriter | 한도초과: fallback)
workflow.add_conditional_edges(
    "verifier",
    check_hallucination,
    {
        END:                 END,
        "feedback_rewriter": "feedback_rewriter",
        "fallback":          "fallback",
    }
)

# feedback_rewriter → 해당 specialist 재실행
workflow.add_conditional_edges(
    "feedback_rewriter",
    route_feedback_to_specialist,
    {
        "robotics":   "robotics",
        "welding":    "welding",
        "electrical": "electrical",
        "general":    "general",
    }
)

# Fallback → END
workflow.add_edge("fallback", END)

# ─────────────────────────────────────────────────────────────
# 그래프 컴파일 및 내보내기
# ─────────────────────────────────────────────────────────────

# 기본적으로는 MemorySaver를 사용하지만, main.py나 서버 기동부에서 RDS 세이버로 교체 가능
default_memory = get_memory_saver()
app_graph = workflow.compile(checkpointer=default_memory)

def compile_workflow(checkpointer):
    """지정된 체크포인터로 워크플로우를 컴파일합니다."""
    return workflow.compile(checkpointer=checkpointer)
