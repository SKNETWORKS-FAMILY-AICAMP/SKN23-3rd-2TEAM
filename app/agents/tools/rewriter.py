# ============================================================
# [쿼리 재작성 노드 — 최종 완성판]
# app/agents/tools/rewriter.py
# ============================================================
# 3단계 하이브리드 전처리 엔진:
#   1단계: JARGON_MAP  — Python dict O(1) 치환 (무비용)
#   2단계: BRAND_CODE_MAP — Regex 에러코드→브랜드 자동 추론 (무비용)
#   3단계: gpt-4o-mini — 검색 최적화 쿼리 확장 (저비용)
#
# RAW_DATA 실제 폴더/파일 구조(구글 드라이브) 기반:
#   ABB_robot/      → IRC5, IRB1600/2400/2600/4600
#   Doosan_robot/   → A-Series, MH-Series (V2.10, V3.4)
#   HD_robot/       → HH/HA/HC/HDR 시리즈, Hi5/Hi6, TP630
#   RB_robot/       → RB 시리즈 (레인보우로보틱스)
#   UR_robot/       → UR3/UR10e/UR20/UR30, e-Series
#   Yaskawa_robot/  → AR700/1440/1730/2010, YRC1000micro
# ============================================================
import re
import csv
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from app.schemas.state import GraphState


# ─────────────────────────────────────────────────────────────
# [1단계] 현장 은어 사전 — JARGON_MAP (O(1) 검색 최적화)
#   CSV가 없으면 이 built-in 딕셔너리가 fallback으로 사용됩니다.
# ─────────────────────────────────────────────────────────────
_BUILTIN_JARGON_MAP: dict[str, str] = {
    # ── 로봇 동작 이상 ──────────────────────────────────────
    "떴어":           "알람(Alarm) 발생",
    "에러":           "알람(Alarm)/에러코드 발생",
    "알람":           "알람(Alarm) 코드 발생",
    "먹통":           "응답 없음(No Response) / 시스템 정지",
    "팅김":           "비정상 종료 / 리셋(Reset) 발생",
    "삑소리":         "알람(Alarm) 음향 경보",
    "떨림":           "진동(Vibration) 이상",
    "덜덜":           "진동(Vibration) / 떨림",
    "멈춰":           "축(Axis) 정지 / 비상정지(E-Stop)",
    "끊겼어":         "케이블 단선 / 통신 오류",
    "원점":           "원점 복귀(Home Position) 보정 / 캘리브레이션",
    "배터리":         "엔코더 배터리(Encoder Battery) 전압 저하",
    "리셋":           "알람 리셋(Alarm Reset) 절차",
    "떨어졌어":       "전압 강하(Voltage Drop) / 연결 해제",
    "절었어":         "모터 탈조(Step-out) / 스텝 손실 발생",   # [NEW] 탈조
    "절어":           "모터 탈조(Step-out) / 스텝 손실 발생",
    # ── 용접 결함 ───────────────────────────────────────────
    "불똥":           "스패터(Spatter) 과다 발생",
    "지직":           "아크 불안정(Arc Instability)",
    "지직거림":       "아크 불안정(Arc Instability) / 아크 단절",
    "펑":             "아크 폭발(Arc Explosion) / 과전류",
    "구멍":           "용락(Burn-Through) 발생",
    "볼록":           "오버랩(Overlap) / 볼록 비드(Convex Bead)",
    "파임":           "언더컷(Undercut) 발생",
    "기포":           "기공(Porosity) 발생",
    "갈라짐":         "균열(Crack) 발생",
    "안붙어":         "융합 불량(Lack of Fusion) / 용입 부족",
    "와이어":         "용접 와이어(Welding Wire) 송급 불량",
    "팁":             "콘택트 팁(Contact Tip) 막힘/마모",
    "가스":           "실드 가스(Shielding Gas) 유량 부족",
    # ── 전기/전장 ────────────────────────────────────────────
    "안들어와":       "입력 전원(Input Power) 차단 / MCCB 트립",
    "전기 안들어옴":  "입력 전원(Input Power) 차단기(MCCB) 점검",
    "차단기":         "배선용 차단기(MCCB) / 누전차단기(ELB)",
    "트립":           "차단기(MCCB) 과전류/누전 트립",
    "두꺼비":         "배선용 차단기(MCCB)",
    "센서 반응":      "센서(Sensor) 오감지(Mis-detection) / 감도 조정",
    "노이즈":         "전자기 간섭(EMI) 노이즈 필터링",
    "접지":           "접지(Grounding) 불량 / 누전",
    "타버림":         "릴레이(Relay) / 차단기(MCCB) 소손(Burnout)",
    # ── 공구/장비 현장 은어 [NEW] ────────────────────────────
    "임팩":           "임팩트 렌치(Impact Wrench)",
    "임팩트":         "임팩트 렌치(Impact Wrench)",
    "구라인다":       "그라인더(Angle Grinder)",
    "그라인다":       "그라인더(Angle Grinder)",
    "노기스":         "캘리퍼스(Vernier Caliper)",
    "노기스로":       "캘리퍼스(Vernier Caliper)로",
    "솔밸브":         "솔레노이드 밸브(Solenoid Valve)",
    "솔레노이드":     "솔레노이드 밸브(Solenoid Valve)",
    "에어건":         "에어 블로우건(Air Blow Gun)",
    "토크렌치":       "토크 렌치(Torque Wrench)",
    "오링":           "O링(O-Ring) 씰(Seal)",
    # ── 안전/작업 용어 ───────────────────────────────────────
    "비계":           "비계(Scaffold) 작업대",
    "락아웃":         "잠금/태깅(LOTO: Lockout-Tagout) 안전 절차",
    "로토":           "LOTO(Lockout-Tagout) 안전 절차",
}

# ─────────────────────────────────────────────────────────────
# CSV 로더 (configs/jargon_map.csv — 관리자 편집 가능)
# ─────────────────────────────────────────────────────────────
JARGON_MAP_CSV = Path(__file__).resolve().parents[3] / "configs" / "jargon_map.csv"

def _load_jargon_map() -> dict[str, str]:
    """
    configs/jargon_map.csv에서 은어 사전을 로드합니다.
    파일이 없거나 오류 시 _BUILTIN_JARGON_MAP을 fallback으로 사용합니다.
    """
    jargon = dict(_BUILTIN_JARGON_MAP)  # built-in을 기본값으로 복사
    try:
        with open(JARGON_MAP_CSV, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            count = 0
            for row in reader:
                slang    = row.get("현장_은어", "").strip()
                standard = row.get("표준_기술_용어", "").strip()
                if slang and standard:
                    jargon[slang] = standard  # CSV가 built-in을 덮어씀
                    count += 1
        print(f"[Rewriter] JARGON_MAP: built-in {len(_BUILTIN_JARGON_MAP)}개 "
              f"+ CSV {count}개 (합계 {len(jargon)}개)")
    except FileNotFoundError:
        print(f"[Rewriter] jargon_map.csv 없음 — built-in {len(jargon)}개 사용")
    except Exception as e:
        print(f"[Rewriter] CSV 로드 오류: {e} — built-in 사용")
    return jargon

# 서버 시작 시 1회 로드 (모듈 임포트 시점)
JARGON_MAP: dict[str, str] = _load_jargon_map()


# ─────────────────────────────────────────────────────────────
# [2단계] 6대 브랜드 에러코드/모델명 자동 추론 (RAW_DATA 실제 구조 반영)
# ─────────────────────────────────────────────────────────────
BRAND_CODE_MAP = [
    # ── HD 현대로보틱스 (HH/HA/HC/HDR 시리즈, Hi5/Hi6 제어기) ──
    (r"\bE0\d{2}\b",                   "현대로보틱스(HD) Hi5/Hi6 제어기"),
    (r"\bE[1-9]\d{3}\b",               "현대로보틱스(HD) Hi6 제어기"),
    (r"\b(Hi5|Hi6|TP630|HDR|HH\d+|HA\d+|HC\d+)\b",
                                       "현대로보틱스(HD)"),
    # ── Yaskawa 야스카와 (AR700/1440/1730/2010, YRC1000micro) ──
    (r"\b41\d{2}\b",                   "야스카와(Yaskawa) YRC1000micro 제어기 알람코드"),
    (r"\b(YRC1000|YRC|DX200|AR700|AR1440|AR1730|AR2010)\b",
                                       "야스카와(Yaskawa)"),
    # ── Doosan 두산로보틱스 (A-Series, MH-Series) ──
    (r"\bM0\d{2}\b",                   "두산로보틱스(Doosan) 협동로봇"),
    (r"\b(Doosan|두산|A-Series|MH-Series|doosan)\b",
                                       "두산로보틱스(Doosan)"),
    # ── ABB (IRC5, IRB 1600/2400/2600/4600) ──
    (r"\b3HAC\d+\b",                   "ABB IRC5 제어기"),
    (r"\b(IRC5|IRB\s?\d{3,4}|ABB)\b",  "ABB"),
    # ── UR 유니버설로봇 (UR3/UR10e/UR20/UR30, e-Series) ──
    (r"\b(UR3|UR10e?|UR20|UR30|UR5|e-Series|polyscope)\b",
                                       "유니버설로봇(UR Universal Robots)"),
    # ── RB 레인보우로보틱스 (RB5/RB10/RB16) ──
    (r"\b(RB5|RB10|RB16|RB\d+|레인보우)\b",
                                       "레인보우로보틱스(RB Rainbow Robotics)"),
]


# ─────────────────────────────────────────────────────────────
# 공통 전처리 유틸
# ─────────────────────────────────────────────────────────────
def normalize_jargon(text: str) -> str:
    """
    JARGON_MAP → BRAND_CODE_MAP 순서로 1차 정규화합니다.
    LLM 호출 없이 즉시 실행됩니다. (O(n) 치환)
    """
    normalized = text
    # 1단계: 은어 치환
    for slang, standard in JARGON_MAP.items():
        normalized = re.sub(re.escape(slang), standard, normalized, flags=re.IGNORECASE)
    # 2단계: 브랜드 에러코드 추론
    for pattern, brand_keyword in BRAND_CODE_MAP:
        if re.search(pattern, normalized, flags=re.IGNORECASE):
            if brand_keyword not in normalized:
                normalized = f"{brand_keyword} {normalized}"
    return normalized.strip()


# ─────────────────────────────────────────────────────────────
# [3단계] LLM 쿼리 확장 프롬프트
# ─────────────────────────────────────────────────────────────
QUERY_REWRITER_PROMPT = """당신은 산업 현장 기술 문서(로봇/용접기/전기 배선) 전문 검색 쿼리 최적화 전문가입니다.
[1차 정규화된 질문]과 [대화 맥락]을 바탕으로, 아래 6대 브랜드 매뉴얼 DB에서
가장 정확한 문서가 나오도록 '확장된 기술 검색 쿼리'를 생성하세요.

[지원 브랜드 및 주요 모델]
- 현대로보틱스(HD): HH/HA/HC/HDR 시리즈, Hi5/Hi6 제어기, TP630
- 야스카와(Yaskawa): AR700/AR1440/AR1730/AR2010, YRC1000micro 제어기
- 두산로보틱스(Doosan): A-Series, MH-Series, V2.10/V3.4
- ABB: IRB1600/2400/2600/4600, IRC5/IRC5 Compact 제어기
- 유니버설로봇(UR): UR3/UR10e/UR20/UR30, e-Series, PolyScope
- 레인보우로보틱스(RB): RB5/RB10/RB16 시리즈

[변환 원칙]
1. 원본 질문의 핵심 의도(에러코드/증상/브랜드)를 반드시 유지하세요.
2. 제조사+모델+에러코드를 구체적으로 포함하세요.
3. 증상(Symptom), 원인(Cause), 조치(Action) 키워드를 모두 포함하세요.
4. 한국어와 영문 기술 용어를 병행하세요. (예: "스패터(Spatter)")
5. 질문이 완전한 일상 대화(날씨/정치 등)이면 "[GENERAL]"이라고만 출력하세요.
6. 검색 쿼리 한 줄만 출력, 부가 설명 없음.

[변환 예시]
입력: "야스카와(Yaskawa) YRC1000micro 알람코드 41XX"
출력: "야스카와 Yaskawa YRC1000micro YRC1000 제어기 알람 4107 서보 드라이브 전류 이상 원인 조치"

입력: "현대로보틱스(HD) Hi6 E012 에러 배터리"
출력: "현대로보틱스 HD Hi5 Hi6 TP630 E012 서보 엔코더 배터리(ER6VC119A) 교체 알람 리셋 절차"

입력: "탄소강 MAG 용접 스패터(Spatter) 과다 발생"
출력: "탄소강 MAG 용접 스패터 Spatter 과다 발생 원인 전압 와이어 가스 유량 콘택트 팁 조치"

입력: "임팩트 렌치(Impact Wrench) 볼트 체결 토크"
출력: "임팩트 렌치 Impact Wrench 볼트 체결 조임 토크 Nm 기준값 관리 방법"

[대화 맥락]
{chat_history}

[1차 정규화된 질문]
{normalized_query}

검색 쿼리를 출력하세요:"""


# ─────────────────────────────────────────────────────────────
# 피드백 재작성 프롬프트
# ─────────────────────────────────────────────────────────────
FEEDBACK_REWRITER_PROMPT = """당신은 산업 현장 기술 문서 검색 전문가입니다.
이전 검색이 실패했습니다. Verifier 실패 분석을 바탕으로 쿼리를 재정교화하세요.

⚠️ [핵심 원칙: 원본 의도 보존]
피드백 루프를 반복할수록 쿼리가 원래 의도에서 벗어날 위험이 있습니다.
아래 [사용자 원본 질문]을 항상 대조군으로 참조하세요.
개선된 쿼리는 원본 질문의 핵심 의도(에러코드/증상/브랜드)를 반드시 유지해야 합니다.

[사용자 원본 질문 — 절대 이탈 금지]
{original_question}

[실패한 기존 검색 쿼리]
{prev_query}

[Verifier 실패 분석]
{feedback}

[재작성 규칙]
1. 누락된 키워드(에러코드, 모델명, 부품명, 수치)를 추가하세요.
2. 범위가 너무 넓었다면 → 브랜드/모델을 더 구체적으로 좁히세요.
3. 결과가 없었다면 → 유의어, 영문 표기, 알람코드 변형을 추가하세요.
4. 원본 의도에서 벗어난 키워드는 추가하지 마세요.
5. 개선된 검색 쿼리 한 줄만 출력, 부가 설명 없이.

개선된 검색 쿼리:"""


# ─────────────────────────────────────────────────────────────
# Public 유틸 함수
# ─────────────────────────────────────────────────────────────
def rewrite_query(original_query: str, chat_history: str = "") -> str:
    """
    현장 작업자의 짧은 질문을 3단계로 처리합니다.
      1단계 — JARGON_MAP + BRAND_CODE_MAP 정규화 (무비용)
      2단계 — gpt-4o-mini 기술 쿼리 확장
    """
    normalized = normalize_jargon(original_query)
    print(f"[Rewriter] 원본:   '{original_query}'")
    print(f"[Rewriter] 정규화: '{normalized}'")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", QUERY_REWRITER_PROMPT),
        ("human", "변환해주세요."),
    ])
    chain = prompt | llm
    result = chain.invoke({
        "normalized_query": normalized,
        "chat_history":     chat_history or "없음",
    })
    rewritten = result.content.strip()
    print(f"[Rewriter] 확장:   '{rewritten}'")
    return rewritten


# ─────────────────────────────────────────────────────────────
# LangGraph 노드 (async — LangGraph async 호환)
# ─────────────────────────────────────────────────────────────
async def rewriter_node(state: GraphState) -> dict:
    """
    LangGraph 쿼리 재작성 노드 (async).

    [역할]
      - 사용자의 첫 질문을 수신하여 3단계 쿼리 확장 수행
      - original_question을 State에 고정 저장 (피드백 루프 내 의도 보존 앵커)
      - rewritten_query를 State에 업데이트
    """
    print("--- [Node: Rewriter] 쿼리 최적화 중 ---")
    messages = state.get("messages", [])
    if not messages:
        return {"rewritten_query": "", "original_question": ""}

    original_query = messages[-1].content
    # [원본 질문 앵커] 피드백 루프 재진입 시에도 최초 질문을 유지
    original_question = state.get("original_question") or original_query

    # 모든 이전 메시지를 컨텍스트로 사용
    history_msgs = messages[:-1]
    chat_history = "\n".join([
        f"{'사용자' if msg.type == 'human' else 'AI'}: {msg.content}"
        for msg in history_msgs
    ]) if history_msgs else ""

    rewritten = rewrite_query(original_query, chat_history)
    return {
        "rewritten_query":   rewritten,
        "original_question": original_question,
    }


async def feedback_rewriter_node(state: GraphState) -> dict:
    """
    Verifier 실패 후 피드백 기반 쿼리 재정교화 노드 (async).

    [원본 의도 보존 설계]
    - messages[0] (대화의 첫 번째 메시지) = 사용자의 최초 원본 질문
    - state.original_question에도 저장되어 있으나 messages[0]을 최종 대조군으로 사용
    - 피드백 루프를 반복해도 원래 의도에서 벗어나지 않음

    [반환]
    - rewritten_query: 개선된 검색 쿼리
    - verifier_feedback: "" (소비 후 초기화 — 무한루프 방지)
    """
    print("--- [Node: FeedbackRewriter] 피드백 기반 쿼리 재정교화 ---")

    prev_query  = state.get("rewritten_query", "")
    feedback    = state.get("verifier_feedback", "")
    messages    = state.get("messages", [])
    retry_count = state.get("retry_count", 0)

    if not feedback:
        print("[FeedbackRewriter] 피드백 없음 — 기존 쿼리 유지")
        return {"verifier_feedback": ""}

    # [원본 질문 앵커] messages[0] → 대화 최초 질문 (의도 대조군)
    original_question = (
        state.get("original_question")          # State 저장값 우선
        or (messages[0].content if messages else "")   # fallback: messages[0]
    )

    print(f"[FeedbackRewriter] retry_count: {retry_count}")
    print(f"[FeedbackRewriter] 원본 질문 앵커: '{original_question}'")
    print(f"[FeedbackRewriter] 기존 쿼리: '{prev_query[:80]}'")
    print(f"[FeedbackRewriter] 피드백:   '{feedback[:100]}'")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", FEEDBACK_REWRITER_PROMPT),
        ("human", "쿼리를 재작성해주세요."),
    ])
    chain = prompt | llm
    result = await chain.ainvoke({
        "original_question": original_question,
        "prev_query":        prev_query,
        "feedback":          feedback,
    })
    refined_query = result.content.strip()
    print(f"[FeedbackRewriter] 개선된 쿼리: '{refined_query}'")

    return {
        "rewritten_query":   refined_query,
        "verifier_feedback": "",  # [중요] 소비 후 초기화 → 무한루프 방지
        # original_question은 변경하지 않음 (앵커 유지)
    }
