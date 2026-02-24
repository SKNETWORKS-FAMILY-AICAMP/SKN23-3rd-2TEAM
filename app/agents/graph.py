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
from app.agents.tools.rewriter import rewriter_node, feedback_rewriter_node
from app.core.security import verifier_node

# ── Fallback 로그 경로 (① 관리자 로그 수집) ──────────────────
FALLBACK_LOG_PATH = Path(__file__).resolve().parents[2] / "logs" / "fallback_queries.jsonl"

FALLBACK_MESSAGE = (
    "현재 시스템의 매뉴얼 및 DB에서는 안전하고 정확한 해결책을 찾을 수 없습니다. "
    "억측으로 인한 설비 파손 및 안전사고를 방지하기 위해, "
    "현대로보틱스 고객센터(1588-4415)나 원본 매뉴얼을 직접 확인해 주십시오."
)

def fallback_node(state: GraphState) -> dict:
    """
    Verifier 최대 재시도(retry_count >= 2) 또는 도메인 재분류 실패 후 발동하는 안전망.

    ① [관리자 로그 수집] fallback으로 들어온 질문을 JSONL로 기록합니다.
      → 관리자가 매주 이 파일을 보고 누락된 매뉴얼을 파악/업로드할 수 있습니다.
      → 저장 경로: logs/fallback_queries.jsonl
    """
    messages    = state.get("messages", [])
    question    = state.get("original_question") or (messages[-1].content if messages else "")
    category    = state.get("category","?")
    retry_count = state.get("retry_count", 0)
    routing_retry = state.get("routing_retry", 0)

    print(f"--- [Node: Fallback] ⚠️ 안전망 발동 ---")
    print(f"  질문: '{question[:60]}'  category={category}  "
          f"retry={retry_count}  routing_retry={routing_retry}")

    # ① 관리자용 JSONL 로그 수집
    try:
        FALLBACK_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        log_entry = {
            "timestamp":     datetime.now().isoformat(),
            "question":      question,
            "category":      category,
            "retry_count":   retry_count,
            "routing_retry": routing_retry,
            "rewritten_query": state.get("rewritten_query",""),
            "verifier_feedback": state.get("verifier_feedback",""),
        }
        with open(FALLBACK_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        print(f"  [Fallback] 📝 로그 기록 완료: {FALLBACK_LOG_PATH}")
    except Exception as e:
        print(f"  [Fallback] ⚠️ 로그 기록 실패: {e}")

    return {
        "generated_answer":  FALLBACK_MESSAGE,
        "is_hallucinated":   False,
        "verifier_feedback": "",
        "domain_mismatch":   False,
    }

def route_to_specialist(state: GraphState) -> str:
    """Supervisor의 category 결정에 따라 specialist를 선택합니다."""
    return state.get("category", "general")

def check_domain_mismatch(state: GraphState) -> str:
    """
    specialist가 domain_mismatch=True를 반환했을 때의 라우팅 결정.

    ③ [무한루프 방지] routing_retry >= 1이면 fallback으로 진행합니다.
        같은 질문이 계속 라우팅 실패를 반복하는 루프를 1회로 제한.
    """
    if not state.get("domain_mismatch", False):
        return "verifier"  # 정상 → 검증 단계로

    routing_retry = state.get("routing_retry", 0)
    if routing_retry >= 1:
        print(f"[DomainGuard] ⚠️ routing_retry={routing_retry} ≥ 1 → fallback (무한루프 방지)")
        return "fallback"

    category = state.get("category","?")
    print(f"[DomainGuard] 🔄 도메인 불일치 (1회 허용) → supervisor 재분류 (현재 category={category})")
    return "supervisor_reroute"  # supervisor로 재분류 요청

async def reroute_supervisor_node(state: GraphState) -> dict:
    """
    domain_mismatch 발생 시 supervisor를 재호출하여 도메인을 재분류합니다 (async).
    """
    print("--- [Node: SupervisorReroute] 도메인 재분류 시도 ---")
    routing_retry = state.get("routing_retry", 0)
    # supervisor_node (async) 호출
    result = await supervisor_node(state)  # [FIX] await 추가
    result["routing_retry"]  = routing_retry + 1
    result["domain_mismatch"] = False  # 초기화
    old_cat = state.get("category","?")
    new_cat = result.get("category","?")
    print(f"  [Reroute] {old_cat} → {new_cat}  (routing_retry={routing_retry+1})")
    return result

def check_hallucination(state: GraphState) -> str:
    """
    Verifier 결과 기반 라우팅.

    ✅ 통과             → END
    🔄 실패 + retry<2   → feedback_rewriter (피드백 기반 재정교화)
    ❌ 실패 + retry≥2   → fallback (안전망)
    """
    is_hallucinated = state.get("is_hallucinated", False)
    retry_count     = state.get("retry_count", 0)

    if not is_hallucinated:
        print(f"[check_hallucination] ✅ 통과 → END")
        return END

    if retry_count < 2:
        print(f"[check_hallucination] 🔄 실패 (retry={retry_count}) → feedback_rewriter")
        return "feedback_rewriter"

    print(f"[check_hallucination] ❌ 한도 초과 (retry={retry_count}) → fallback")
    return "fallback"

def route_feedback_to_specialist(state: GraphState) -> str:
    """feedback_rewriter 완료 후 원래 도메인 specialist로 돌아갑니다."""
    return state.get("category", "general")

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
workflow.add_node("verifier",           verifier_node)
workflow.add_node("feedback_rewriter",  feedback_rewriter_node)
workflow.add_node("fallback",           fallback_node)
workflow.add_node("supervisor_reroute", reroute_supervisor_node)  # ③ 도메인 재분류

# 기본 흐름: START → rewriter → supervisor
workflow.add_edge(START, "rewriter")
workflow.add_edge("rewriter", "supervisor")

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

# general → 검증 없이 END
workflow.add_edge("general", END)

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

memory = get_memory_saver()
app_graph = workflow.compile(checkpointer=memory)
