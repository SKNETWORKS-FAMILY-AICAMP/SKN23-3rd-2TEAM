# ============================================================
# [쿼리 재작성 노드 — 최종 완성판]
# app/agents/tools/rewriter.py
# ============================================================
# 3단계 하이브리드 전처리 엔진:
#   1단계: JARGON_MAP  — AWS RDS 동기화 (O(1) 치환)
#   2단계: BRAND_CODE_MAP — Regex 에러코드→브랜드 자동 추론 (단어 경계 엄격 적용)
#   3단계: gpt-4o — 검색 최적화 쿼리 확장 (과대 추론 방지 적용)
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
import threading
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from app.schemas.state import GraphState
from app.core.config import MODEL_FAST

# ─────────────────────────────────────────────────────────────
# [1단계] 현장 은어 사전 — JARGON_MAP (O(1) 검색 최적화)
#   AWS RDS 연결 실패 시에만 이 built-in 딕셔너리가 fallback으로 사용됩니다.
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
    "절었어":         "모터 탈조(Step-out) / 스텝 손실 발생",
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
    # ── 공구/장비 현장 은어 ────────────────────────────
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
# AWS RDS 로더 (jargon_map 테이블 전용)
# ─────────────────────────────────────────────────────────────
def _load_jargon_map() -> dict[str, str]:
    """
    AWS RDS의 jargon_map 테이블에서 은어 사전을 로드합니다.
    DB 연결 실패 시 _BUILTIN_JARGON_MAP을 fallback으로 사용합니다.
    """
    jargon = dict(_BUILTIN_JARGON_MAP)
    import psycopg2
    import os
    
    try:
        ssh_enabled = os.getenv("SSH_TUNNEL_ENABLED", "false").lower() == "true"
        ssh_local_port = os.getenv("SSH_LOCAL_BIND_PORT", "15432")
        target_host = "127.0.0.1" if ssh_enabled else os.getenv("PGHOST", "localhost")
        target_port = ssh_local_port if ssh_enabled else os.getenv("PGPORT", "5432")
        
        conn = psycopg2.connect(
            host=target_host,
            database=os.getenv("PGDATABASE", "chatbot_db"),
            user=os.getenv("PGUSER", "postgres"),
            password=os.getenv("PGPASSWORD", "password"),
            port=target_port,
            connect_timeout=5
        )
        
        with conn.cursor() as cur:
            cur.execute("SELECT jargon, standard FROM jargon")
            rows = cur.fetchall()
            count = 0
            for slang, standard in rows:
                if slang and standard:
                    jargon[slang.strip()] = standard.strip()
                    count += 1
            print(f"[Rewriter] RDS JARGON 로드 완료: {count}개 항목 추가됨.")
        conn.close()
    except Exception as e:
        print(f"[Rewriter] RDS JARGON_MAP 로드 실패: {e} — built-in {len(_BUILTIN_JARGON_MAP)}개 사용")
        
    return jargon

_JARGON_MAP_CACHE: dict[str, str] | None = None
_JARGON_MAP_LOCK = threading.Lock()

def get_jargon_map() -> dict[str, str]:
    """
    Delay DB-backed jargon loading until first use.
    This avoids import-time connection attempts before SSH tunnel startup.
    """
    global _JARGON_MAP_CACHE
    if _JARGON_MAP_CACHE is not None:
        return _JARGON_MAP_CACHE

    with _JARGON_MAP_LOCK:
        if _JARGON_MAP_CACHE is None:
            _JARGON_MAP_CACHE = _load_jargon_map()
    return _JARGON_MAP_CACHE


# ─────────────────────────────────────────────────────────────
# [2단계] 6대 브랜드 에러코드/모델명 자동 추론 (엄격한 단어 경계 적용)
# ─────────────────────────────────────────────────────────────
BRAND_CODE_MAP = [
    # ── HD 현대로보틱스 ──
    (r"\bE0\d{2}\b",                   "현대로보틱스(HD) Hi5/Hi6 제어기"),
    (r"\bE[1-9]\d{3}\b",               "현대로보틱스(HD) Hi6 제어기"),
    (r"\b(Hi5|Hi6|TP630|HDR|HH\d+|HA\d+|HC\d+)\b", "현대로보틱스(HD)"),
    # ── Yaskawa 야스카와 ──
    (r"\b41\d{2}\b",                   "야스카와(Yaskawa) YRC1000micro 제어기 알람코드"),
    (r"\b(YRC1000|YRC|DX200|AR700|AR1440|AR1730|AR2010)\b", "야스카와(Yaskawa)"),
    # ── Doosan 두산로보틱스 ──
    (r"\bM0\d{2}\b",                   "두산로보틱스(Doosan) 협동로봇"),
    (r"\b(Doosan|두산|A-Series|MH-Series|doosan)\b", "두산로보틱스(Doosan)"),
    # ── ABB ──
    (r"\b3HAC\d+\b",                   "ABB IRC5 제어기"),
    (r"\b(IRC5|IRB\s?\d{3,4}|ABB)\b",  "ABB"),
    # ── UR 유니버설로봇 ──
    (r"\b(UR3|UR10e?|UR20|UR30|UR5|e-Series|polyscope|유알|유니버설로봇?)\b", "유니버설로봇(UR Universal Robots)"),
    # ── RB 레인보우로보틱스 ──
    (r"\b(RB5|RB10|RB16|RB\d+|레인보우(?:로보틱스)?)\b", "레인보우로보틱스(RB Rainbow Robotics)"),
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
    for slang, standard in get_jargon_map().items():
        normalized = re.sub(re.escape(slang), standard, normalized, flags=re.IGNORECASE)
    # 2단계: 브랜드 에러코드 추론
    for pattern, brand_keyword in BRAND_CODE_MAP:
        if re.search(pattern, normalized, flags=re.IGNORECASE):
            if brand_keyword not in normalized:
                normalized = f"{brand_keyword} {normalized}"
    return normalized.strip()


# ─────────────────────────────────────────────────────────────
# [3단계] LLM 쿼리 확장 프롬프트 (과대 추론 방지 조항 추가)
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
1. [과대 추론 금지] 사용자의 원본 질문에 특정 브랜드(현대, 야스카와, 두산 등)나 모델명이 명시적으로 존재하지 않는다면, 단순 오타나 깨진 글자(예: '혀')를 보고 임의로 특정 브랜드를 추측하여 쿼리에 추가하지 마십시오.
2. 포괄적인 질문("로봇 브랜드 알려줘", "용접기 추천해줘" 등)은 특정 브랜드로 좁히지 말고, 포괄적인 의미를 그대로 유지하여 정규화하십시오.
3. 원본 질문의 핵심 의도(에러코드/증상/브랜드)를 반드시 유지하세요.
4. 제조사+모델+에러코드를 구체적으로 포함하세요.
5. 증상(Symptom), 원인(Cause), 조치(Action) 키워드를 모두 포함하세요.
6. 한국어와 영문 기술 용어를 병행하세요. (예: "스패터(Spatter)")
7. **[중요] 특정 로봇 브랜드(현대, 레인보우 등)에 대한 궁금증이나 언급이 있다면 일상 대화가 아닌 기술 질의로 간주하여 검색 쿼리를 생성하세요.**
8. 질문이 완전히 기술/브랜드와 무관한 일상 대화(날씨/정치 등)인 경우에만 "[GENERAL]"이라고 출력하세요.
9. 검색 쿼리 한 줄만 출력, 부가 설명 없음.

[변환 예시]
입력: "야스카와(Yaskawa) YRC1000micro 알람코드 41XX"
출력: "야스카와 Yaskawa YRC1000micro YRC1000 제어기 알람 4107 서보 드라이브 전류 이상 원인 조치"

입력: "현대로보틱스(HD) Hi6 E012 에러 배터리"
출력: "현대로보틱스 HD Hi5 Hi6 TP630 E012 서보 엔코더 배터리(ER6VC119A) 교체 알람 리셋 절차"

입력: "혀로봇브랜드 알려줘"
출력: "산업용 다관절 로봇 브랜드 라인업 정보"

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
# [0단계] Fast Track - 소셜 인사 패턴 (Bypass LLM)
# ─────────────────────────────────────────────────────────────
SOCIAL_GREETING_PATTERNS = [
    r"^(안녕\s*|하이\s*|hi\s*|hello\s*|반가워\s*|ㅎㅇ\s*|ㅎㄹ\s*)$",  
    r"^(누구야|이름이 뭐야|뭐하는 애야|당신은 누구|너는 누구)$", 
    r"^(고마워|감사|땡큐|thanks|thx)$",
    r"^(잘가|바이|bye|수고)$",
]

def is_social_greeting(text: str) -> bool:
    """단순 인사나 일상적인 대화인지 정규표현식으로 빠르게 판단합니다."""
    clean_text = text.strip()
    for pattern in SOCIAL_GREETING_PATTERNS:
        if re.search(pattern, clean_text, re.IGNORECASE):
            return True
    return False

# ─────────────────────────────────────────────────────────────
# Public 유틸 함수
# ─────────────────────────────────────────────────────────────
from typing import Tuple

def rewrite_query(original_query: str, chat_history: str = "") -> Tuple[str, str]:
    """
    [V3.4] 쿼리 오염 방지를 위해 (확장쿼리, 라우팅힌트) 튜플을 반환합니다.
      0단계 — Fast Track (인사/일상대화 감지)
      1단계 — JARGON_MAP + BRAND_CODE_MAP 정규화
      2단계 — gpt-4o 기술 쿼리 확장
    """
    current_jargon = get_jargon_map()

    if is_social_greeting(original_query):
        print(f"[Rewriter] 소셜 인사 감지 → Fast Track 발동")
        return original_query, "SOCIAL"

    normalized = original_query
    for slang, standard in current_jargon.items():
        normalized = re.sub(re.escape(slang), standard, normalized, flags=re.IGNORECASE)
    
    for pattern, brand_keyword in BRAND_CODE_MAP:
        if re.search(pattern, normalized, flags=re.IGNORECASE):
            if brand_keyword not in normalized:
                normalized = f"{brand_keyword} {normalized}"

    print(f"[Rewriter] 원본:   '{original_query}'")
    print(f"[Rewriter] 정규화: '{normalized}'")

    llm = ChatOpenAI(model=MODEL_FAST, temperature=0)
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
    
    if "[GENERAL]" in rewritten.upper():
        print(f"[Rewriter] 일반 질문 감지 ([GENERAL]) → Supervisor 라우팅 유도")
        return normalized, "GENERAL"

    print(f"[Rewriter] 확장:   '{rewritten}'")
    return rewritten, "TECHNICAL"


# ─────────────────────────────────────────────────────────────
# LangGraph 노드 (async — LangGraph async 호환)
# ─────────────────────────────────────────────────────────────
async def rewriter_node(state: GraphState) -> dict:
    print("--- [Node: Rewriter] 쿼리 최적화 중 ---")
    messages = state.get("messages", [])
    if not messages:
        return {"rewritten_query": "", "original_question": "", "routing_hint": ""}

    original_query = messages[-1].content
    original_question = state.get("original_question") or original_query

    history_msgs = messages[:-1]
    chat_history = "\n".join([
        f"{'사용자' if msg.type == 'human' else 'AI'}: {msg.content}"
        for msg in history_msgs
    ]) if history_msgs else ""

    rewritten, hint = rewrite_query(original_query, chat_history)
    return {
        "rewritten_query":   rewritten,
        "original_question": original_question,
        "routing_hint":      hint,
    }


async def feedback_rewriter_node(state: GraphState) -> dict:
    print("--- [Node: FeedbackRewriter] 피드백 기반 쿼리 재정교화 ---")

    prev_query  = state.get("rewritten_query", "")
    feedback    = state.get("verifier_feedback", "")
    messages    = state.get("messages", [])
    retry_count = state.get("retry_count", 0)

    if not feedback:
        print("[FeedbackRewriter] 피드백 없음 — 기존 쿼리 유지")
        return {"verifier_feedback": ""}

    original_question = (
        state.get("original_question")
        or (messages[0].content if messages else "")
    )

    print(f"[FeedbackRewriter] retry_count: {retry_count}")
    print(f"[FeedbackRewriter] 원본 질문 앵커: '{original_question}'")
    print(f"[FeedbackRewriter] 기존 쿼리: '{prev_query[:80]}'")
    print(f"[FeedbackRewriter] 피드백:   '{feedback[:100]}'")

    llm = ChatOpenAI(model=MODEL_FAST, temperature=0)
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
        "verifier_feedback": "", 
    }
