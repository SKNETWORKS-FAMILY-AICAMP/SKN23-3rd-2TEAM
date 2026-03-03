from typing import TypedDict, Annotated, Sequence, Dict, Any
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class GraphState(TypedDict):
    """
    LangGraph에서 노드 간 데이터를 주고받는 상태(State) 딕셔너리.
    모든 노드가 이 상태를 공유하며 업데이트합니다.
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]

    category: str              # supervisor 분류 결과 (robotics/welding/electrical/general)
    extracted_model: str       # 메타데이터 필터용 모델명 (예: Hi6)
    rewritten_query: str       # Rewriter가 확장한 검색 최적화 쿼리
    routing_hint: str          # [V3.4] Rewriter가 판단한 라우팅 힌트 (SOCIAL/GENERAL 등)
    original_question: str     # 사용자의 원본 질문 — 피드백 루프 내 의도 보존
    context: str               # RAG 검색 결과 (문서 청크 텍스트)
    generated_answer: str      # Specialist가 생성한 답변
    is_hallucinated: bool      # 환각 여부 (True=실패)
    reranker_score: float      # 검색 점수 최고값 모니터링
    verifier_passed: bool      # 환각 검증 통과 여부 모니터링
    retry_count: int           # Verifier 재시도 횟수 (최대 2 → fallback)
    verifier_feedback: str     # Verifier 실패 분석 → feedback_rewriter에 전달
    domain_mismatch: bool      # specialist가 도메인 불일치 감지 시 True
    routing_retry: int         # domain_mismatch 재분류 허용 횟수 (최대 1)
    metadata: Dict[str, Any]   # [V3.1] 실행 통계 (응답 시간, 장치 가속 등) -> RDS 영구 저장