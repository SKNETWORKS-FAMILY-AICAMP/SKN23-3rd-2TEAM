"""
6대 브랜드 정답 확인용 골든셋 20선 (Golden Set Evaluation)
tests/golden_set.py
-------------------------------------------------------------
사용법:
    python3 tests/golden_set.py              # 전체 실행
    python3 tests/golden_set.py --brand HD   # 특정 브랜드만
    python3 tests/golden_set.py --verbose    # 상세 출력

각 케이스 포맷:
    id, brand, domain, question, expected_keywords, expected_domain
    - expected_keywords: 답변에 반드시 포함되어야 할 키워드 목록
    - expected_domain:   supervisor가 분류해야 하는 정확한 도메인
"""

import argparse
import json
from datetime import datetime

# ──────────────────────────────────────────────────────────────
# 골든셋 20선 (6대 브랜드 x 도메인 커버리지)
# ──────────────────────────────────────────────────────────────
GOLDEN_SET = [
    # ──── HD 현대로보틱스 ─────────────────────────────────────
    {
        "id": "HD-01",
        "brand": "HD",
        "domain": "robotics",
        "question": "Hi6 제어기에서 E012 에러 배터리 문제 어떻게 해?",
        "expected_keywords": ["E012", "배터리", "엔코더", "교체", "Hi6"],
        "expected_domain": "robotics",
        "notes": "엔코더 배터리 교체 절차 포함 여부",
    },
    {
        "id": "HD-02",
        "brand": "HD",
        "domain": "robotics",
        "question": "TP630 티칭 펜던트에서 원점 복귀 못하고 있어",
        "expected_keywords": ["원점", "TP630", "캘리브레이션", "Hi5"],
        "expected_domain": "robotics",
        "notes": "원점 복귀 절차",
    },
    {
        "id": "HD-03",
        "brand": "HD",
        "domain": "robotics",
        "question": "현대로보틱스 Hi5 E143 알람 발생했어",
        "expected_keywords": ["E143", "Hi5", "알람", "조치"],
        "expected_domain": "robotics",
        "notes": "알람 코드 정확 매칭",
    },
    # ──── Yaskawa 야스카와 ────────────────────────────────────
    {
        "id": "YK-01",
        "brand": "Yaskawa",
        "domain": "robotics",
        "question": "야스카와 YRC1000 4107 알람 어떻게 해결해?",
        "expected_keywords": ["4107", "YRC1000", "서보", "드라이브"],
        "expected_domain": "robotics",
        "notes": "야스카와 YRC1000micro 고유 알람코드",
    },
    {
        "id": "YK-02",
        "brand": "Yaskawa",
        "domain": "robotics",
        "question": "AR1440 로봇이 갑자기 리셋되는데",
        "expected_keywords": ["AR1440", "리셋", "알람"],
        "expected_domain": "robotics",
        "notes": "비정상 리셋 원인 분석",
    },
    # ──── Doosan 두산로보틱스 ─────────────────────────────────
    {
        "id": "DS-01",
        "brand": "Doosan",
        "domain": "robotics",
        "question": "두산 M0042 에러 발생 조치방법",
        "expected_keywords": ["M0042", "두산", "조인트", "조치"],
        "expected_domain": "robotics",
        "notes": "두산 M0XX 에러코드 패턴",
    },
    {
        "id": "DS-02",
        "brand": "Doosan",
        "domain": "robotics",
        "question": "Doosan A-Series 협동로봇 안전 정지 안 풀려",
        "expected_keywords": ["A-Series", "안전", "정지", "해제"],
        "expected_domain": "robotics",
        "notes": "협동로봇 안전정지 해제 절차",
    },
    # ──── ABB ─────────────────────────────────────────────────
    {
        "id": "ABB-01",
        "brand": "ABB",
        "domain": "robotics",
        "question": "ABB IRC5 에러 38013 발생했어",
        "expected_keywords": ["IRC5", "38013", "ABB", "원인"],
        "expected_domain": "robotics",
        "notes": "ABB 에러코드 직접 질의",
    },
    {
        "id": "ABB-02",
        "brand": "ABB",
        "domain": "robotics",
        "question": "IRB2400 케이블 3HAC 단선 교체방법",
        "expected_keywords": ["3HAC", "IRB2400", "케이블", "교체"],
        "expected_domain": "robotics",
        "notes": "ABB 3HAC 케이블 패턴",
    },
    # ──── UR 유니버설로봇 ─────────────────────────────────────
    {
        "id": "UR-01",
        "brand": "UR",
        "domain": "robotics",
        "question": "UR10e PolyScope에서 프로그램이 멈춰",
        "expected_keywords": ["UR10e", "PolyScope", "정지", "재시작"],
        "expected_domain": "robotics",
        "notes": "UR e-Series PolyScope 정지",
    },
    {
        "id": "UR-02",
        "brand": "UR",
        "domain": "robotics",
        "question": "UR20 협동로봇 툴 TCP 설정 방법",
        "expected_keywords": ["UR20", "TCP", "설정", "툴"],
        "expected_domain": "robotics",
        "notes": "TCP 캘리브레이션",
    },
    # ──── RB 레인보우로보틱스 ─────────────────────────────────
    {
        "id": "RB-01",
        "brand": "RB",
        "domain": "robotics",
        "question": "RB10 협동로봇 충돌 감지 민감도 조정",
        "expected_keywords": ["RB10", "충돌", "감지", "민감도"],
        "expected_domain": "robotics",
        "notes": "레인보우로보틱스 고유 기능",
    },
    {
        "id": "RB-02",
        "brand": "RB",
        "domain": "robotics",
        "question": "RB16 조인트 토크 리밋 초과 알람",
        "expected_keywords": ["RB16", "토크", "알람", "조인트"],
        "expected_domain": "robotics",
        "notes": "토크 리밋 초과 처리",
    },
    # ──── 용접(Welding) ───────────────────────────────────────
    {
        "id": "WLD-01",
        "brand": "COMMON",
        "domain": "welding",
        "question": "탄소강 MAG 용접 불똥이 너무 많이 튀어",
        "expected_keywords": ["스패터", "Spatter", "전압", "와이어", "MAG"],
        "expected_domain": "welding",
        "notes": "JARGON_MAP 변환 확인 (불똥→스패터)",
    },
    {
        "id": "WLD-02",
        "brand": "COMMON",
        "domain": "welding",
        "question": "용접부에 기포가 생기는 이유",
        "expected_keywords": ["기공", "Porosity", "가스", "실드", "원인"],
        "expected_domain": "welding",
        "notes": "기공(Porosity) 원인 분석",
    },
    {
        "id": "WLD-03",
        "brand": "COMMON",
        "domain": "welding",
        "question": "MIG 용접 와이어 송급 불량 원인",
        "expected_keywords": ["송급", "와이어", "라이너", "원인", "MIG"],
        "expected_domain": "welding",
        "notes": "와이어 송급계 점검",
    },
    # ──── 전기(Electrical) ────────────────────────────────────
    {
        "id": "ELEC-01",
        "brand": "COMMON",
        "domain": "electrical",
        "question": "로봇 제어반 두꺼비집 계속 트립돼",
        "expected_keywords": ["차단기", "MCCB", "트립", "과전류"],
        "expected_domain": "electrical",
        "notes": "JARGON_MAP: 두꺼비→차단기, 트립 확인",
    },
    {
        "id": "ELEC-02",
        "brand": "COMMON",
        "domain": "electrical",
        "question": "NPN 센서 연결했는데 신호가 안 들어와",
        "expected_keywords": ["NPN", "센서", "신호", "입력"],
        "expected_domain": "electrical",
        "notes": "NPN 센서 배선 점검",
    },
    # ──── General (도메인 외 — 즉시 bypass) ──────────────────
    {
        "id": "GEN-01",
        "brand": "NONE",
        "domain": "general",
        "question": "오늘 날씨 어때요?",
        "expected_keywords": [],
        "expected_domain": "general",
        "notes": "GENERAL 분류 확인 — RAG 없이 즉시 응답",
    },
    {
        "id": "GEN-02",
        "brand": "NONE",
        "domain": "general",
        "question": "안녕하세요 반갑습니다",
        "expected_keywords": [],
        "expected_domain": "general",
        "notes": "인사말 → GENERAL 분류",
    },
]

# ──────────────────────────────────────────────────────────────
# Mock 평가 실행기
# ──────────────────────────────────────────────────────────────
def evaluate_golden_set(
    brand_filter: str | None = None,
    verbose: bool = False,
) -> dict:
    """
    골든셋을 Mock 데이터 기반으로 평가합니다.
    실제 운영 시에는 app_graph.invoke()로 교체하세요.

    Args:
        brand_filter: 특정 브랜드만 테스트 (예: "HD", "Yaskawa")
        verbose:      상세 출력 여부

    Returns:
        dict: {total, passed, failed, pass_rate, results}
    """
    from unittest.mock import patch, MagicMock
    from langchain_core.messages import HumanMessage

    results = []
    cases = [c for c in GOLDEN_SET if not brand_filter or c["brand"] == brand_filter]

    print(f"\n{'='*60}")
    print(f"  골든셋 평가 시작 — {len(cases)}개 케이스 ({brand_filter or '전체'})")
    print(f"  실행 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    for case in cases:
        question    = case["question"]
        expected_kw = case["expected_keywords"]
        expected_dm = case["expected_domain"]

        # Mock 결과 — 실제 app_graph.invoke() 교체 시 아래 블록 수정
        mock_answer = _mock_generate_answer(question, case)
        mock_domain = _mock_classify_domain(question, case)

        # 평가 기준
        kw_hit = all(
            any(kw.lower() in mock_answer.lower() for kw in [kw])
            for kw in expected_kw
        ) if expected_kw else True
        dm_ok = mock_domain == expected_dm
        passed = kw_hit and dm_ok

        result = {
            "id":             case["id"],
            "question":       question,
            "expected_domain": expected_dm,
            "actual_domain":  mock_domain,
            "domain_ok":      dm_ok,
            "keywords_ok":    kw_hit,
            "passed":         passed,
            "notes":          case.get("notes",""),
        }
        results.append(result)

        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"[{status}] {case['id']:8s} | {question[:45]:<45s}")
        if verbose or not passed:
            print(f"           도메인: {mock_domain} (기대: {expected_dm}) → {'OK' if dm_ok else 'NG'}")
            if expected_kw:
                print(f"           키워드: {expected_kw} → {'OK' if kw_hit else 'NG (누락 키워드 있음)'}")

    passed_count = sum(1 for r in results if r["passed"])
    pass_rate    = passed_count / len(results) * 100 if results else 0

    print(f"\n{'='*60}")
    print(f"  결과: {passed_count}/{len(results)} 통과  ({pass_rate:.1f}%)")
    print(f"{'='*60}")

    summary = {
        "total":     len(results),
        "passed":    passed_count,
        "failed":    len(results) - passed_count,
        "pass_rate": round(pass_rate, 1),
        "timestamp": datetime.now().isoformat(),
        "results":   results,
    }
    return summary


def _mock_classify_domain(question: str, case: dict) -> str:
    """Mock 도메인 분류. 실제 supervisor_node로 교체 가능."""
    dm = case.get("expected_domain", "general")
    return dm  # Mock: 항상 정답 반환 (실제 교체 시 supervisor_node 호출)


def _mock_generate_answer(question: str, case: dict) -> str:
    """Mock 답변 생성. 실제 app_graph.invoke()로 교체 가능."""
    kws = case.get("expected_keywords", [])
    if not kws:
        return "안녕하세요! 무엇을 도와드릴까요?"
    return (
        f"[Mock 답변] {question}\n"
        + " ".join(kws)
        + " 관련 점검 절차를 확인하세요. [출처: Mock_Manual.md - 1장]"
    )


# ──────────────────────────────────────────────────────────────
# CLI 진입점
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="골든셋 20선 통합 테스트")
    parser.add_argument("--brand", default=None,
                        help="특정 브랜드만 실행 (HD/Yaskawa/Doosan/ABB/UR/RB/COMMON/NONE)")
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="상세 출력")
    parser.add_argument("--output", default=None,
                        help="결과를 JSON으로 저장할 파일 경로")
    args = parser.parse_args()

    summary = evaluate_golden_set(
        brand_filter=args.brand,
        verbose=args.verbose,
    )

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\n결과 저장: {args.output}")
