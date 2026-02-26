"""
tests/eval_rag.py
─────────────────────────────────────────────────────────────────
LLM-as-a-Judge 자동 평가 스크립트 (Ragas-inspired, OpenAI 직접 호출)

평가 지표:
  1. Faithfulness (충실도)    : 답변이 Context에만 기반했는가?
  2. Answer Relevance (관련성): 현장 질문에 실질적 해결책을 제시했는가?
  3. Context Precision (정밀도): 올바른 브랜드 문서가 상위 검색되었는가?

사용법:
  cd SKN23-3rd-2TEAM
  python tests/eval_rag.py                  # 전체 5개 케이스 평가
  python tests/eval_rag.py --case 1         # 특정 케이스만 평가
  python tests/eval_rag.py --output report  # JSON 결과 저장
"""
import os, sys, json, argparse, textwrap
from pathlib import Path
from typing import List, Dict
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from unittest.mock import patch

from app.schemas.state import GraphState
from app.agents.graph import app_graph

# ─────────────────────────────────────────────────────────────
# Mock 문서 (테스트 노트북과 동일한 데이터셋)
# ─────────────────────────────────────────────────────────────
MOCK_DOCS = {
    "robotics": [
        Document(
            page_content=(
                "## E012: 서보 모터 엔코더 배터리 전압 저하\n"
                "Hi6 제어기 E012 알람: 배터리 전압 기준치(3.6V) 이하.\n"
                "교체 절차: 전원 차단 → 볼트 분리 → ER6VC119A 교체 → 알람 리셋 → 원점 재보정"
            ),
            metadata={"source":"Hi6_TP630.md","chapter_path":"4장 > 배터리 교체",
                      "domain":"robotics","brand":"HD","model_name":"Hi6"},
        ),
        Document(
            page_content=(
                "## 야스카와 YRC1000micro 알람코드 4107\n"
                "원인: U/V/W 케이블 단선·접촉 불량, 드라이브 전류 제어 이상.\n"
                "조치: 케이블 커넥터 확인 → 상 저항 측정(1.2Ω±10%) → 드라이브 보드 교체"
            ),
            metadata={"source":"YRC1000micro_Alarm.md","chapter_path":"4000번대 알람",
                      "domain":"robotics","brand":"Yaskawa","model_name":"YRC1000"},
        ),
    ],
    "welding": [
        Document(
            page_content=(
                "## 탄소강 MAG 용접 스패터 과다 원인 및 대책\n"
                "| 원인 | 대책 | 적정값 |\n"
                "|---|---|---|\n"
                "| 전압 저하 | 전압 +1~2V 상향 | 24~27V |\n"
                "| 와이어 송급 불안정 | 롤러 압력·팁 점검 | 5~8m/min |\n"
                "| CO₂ 유량 부족 | 유량계 확인 | 15~20L/min |"
            ),
            metadata={"source":"NCS_MAG_Welding.md","chapter_path":"3장 > 스패터",
                      "domain":"welding","brand":"NCS"},
        ),
    ],
    "electrical": [],
}

BRAND_KEYWORDS = {
    "Yaskawa": ["yaskawa","야스카와","4107","yrc1000"],
    "HD":      ["hd","현대","hi5","hi6","e012","tp630"],
    "Doosan":  ["doosan","두산","m042"],
    "ABB":     ["abb","irc5","3hac"],
    "UR":      ["ur","ur10e","ur3","polyscope"],
    "RB":      ["rb","레인보우"],
}

PATCH_TARGETS = [
    "app.agents.specialists.robotics.run_rag_pipeline",
    "app.agents.specialists.welding.run_rag_pipeline",
    "app.agents.specialists.electrical.run_rag_pipeline",
]

def mock_rag(query: str, domain: str, filters: dict = None) -> str:
    q = query.lower()
    docs = MOCK_DOCS.get(domain.lower(), [])
    for brand, kws in BRAND_KEYWORDS.items():
        if any(k in q for k in kws):
            filtered = [d for d in docs if d.metadata.get("brand") == brand]
            docs = filtered if filtered else docs
            break
    return "\n\n---\n\n".join([
        f"[출처: {d.metadata['source']} | {d.metadata['chapter_path']}]\n{d.page_content}"
        for d in docs[:2]
    ]) if docs else "(관련 문서 없음)"

def run_graph(question: str, thread_id: str) -> dict:
    state: GraphState = {
        "messages":         [HumanMessage(content=question)],
        "category":         "", "extracted_model":  "",
        "rewritten_query":  "", "context":          "",
        "generated_answer": "", "is_hallucinated":  False,
        "retry_count":      0,
    }
    with patch(PATCH_TARGETS[0], side_effect=mock_rag), \
         patch(PATCH_TARGETS[1], side_effect=mock_rag), \
         patch(PATCH_TARGETS[2], side_effect=mock_rag):
        return app_graph.invoke(state, config={"configurable": {"thread_id": thread_id}})

# ─────────────────────────────────────────────────────────────
# 평가 데이터셋 (5개 테스트 케이스)
# ─────────────────────────────────────────────────────────────
TEST_CASES = [
    {
        "id":          1,
        "name":        "Brand Logic — Yaskawa 4107",
        "question":    "4107 해결법",
        "expected_brand": "Yaskawa",
        "expected_keywords": ["서보","드라이브","케이블","저항"],
        "domain":      "robotics",
    },
    {
        "id":          2,
        "name":        "Welding Detail — 스패터 원인",
        "question":    "용접할 때 불똥이 너무 많이 튀어",
        "expected_brand": "NCS",
        "expected_keywords": ["전압","와이어","가스","팁"],
        "domain":      "welding",
    },
    {
        "id":          3,
        "name":        "Cross-Brand — E012 HD 필터링",
        "question":    "E012 에러",
        "expected_brand": "HD",
        "expected_keywords": ["배터리","교체","ER6VC119A","리셋"],
        "domain":      "robotics",
    },
    {
        "id":          4,
        "name":        "Hallucination — 커피 타는 법 (Fallback)",
        "question":    "로봇으로 커피 타는 법 알려줘",
        "expected_brand": None,
        "expected_keywords": ["고객센터", "매뉴얼"],
        "domain":      "general",
        "force_fallback": True,
    },
    {
        "id":          5,
        "name":        "General Bypass — 날씨 질문",
        "question":    "오늘 서울 날씨 어때?",
        "expected_brand": None,
        "expected_keywords": ["로봇","용접","전기"],
        "domain":      "general",
    },
]

# ─────────────────────────────────────────────────────────────
# LLM-as-a-Judge 평가 스키마 및 프롬프트
# ─────────────────────────────────────────────────────────────
class EvalResult(BaseModel):
    faithfulness:        float = Field(ge=0.0, le=1.0,
        description="답변이 Context에만 기반했는가? (0.0=완전 환각, 1.0=완전 충실)")
    answer_relevance:    float = Field(ge=0.0, le=1.0,
        description="현장 질문에 실질적 해결책을 제시했는가? (1.0=매우 관련성 높음)")
    context_precision:   float = Field(ge=0.0, le=1.0,
        description="올바른 브랜드/도메인 문서가 상위 검색되었는가?")
    reasoning:           str   = Field(description="평가 근거 (한국어 2~3문장)")

EVAL_PROMPT = """당신은 산업용 AI 챗봇의 품질 평가 전문가입니다.
아래 질문-Context-답변 트리플에 대해 3가지 지표를 0.0~1.0 점수로 평가하세요.

[질문]
{question}

[검색된 Context]
{context}

[생성된 답변]
{answer}

[평가 기준]
- faithfulness(충실도): 답변이 Context에 없는 내용을 만들어냈으면 낮게, Context에만 기반했으면 높게
- answer_relevance(관련성): 현장 작업자가 즉시 활용할 수 있는 실질적 해결책을 제시했으면 높게
- context_precision(정밀도): 검색된 Context가 질문의 브랜드·도메인과 정확히 일치하면 높게

JSON 형식으로만 응답하세요."""

def evaluate_case(case: dict, state: dict) -> dict:
    """단일 테스트 케이스를 LLM으로 평가합니다."""
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    prompt = ChatPromptTemplate.from_messages([("system", EVAL_PROMPT)])
    chain = prompt | llm.with_structured_output(EvalResult)

    question = case["question"]
    context  = state.get("context", "(없음)")
    answer   = state.get("generated_answer", "(없음)")

    # context가 너무 길면 앞 2000자만 사용
    context_trimmed = context[:2000] if len(context) > 2000 else context

    result = chain.invoke({
        "question": question,
        "context":  context_trimmed,
        "answer":   answer,
    })

    # 규칙 기반 context_precision 보정 (expected_brand 확인)
    expected_brand = case.get("expected_brand")
    if expected_brand and context:
        if expected_brand.lower() not in context.lower() and \
           expected_brand not in context:
            result.context_precision = max(0.0, result.context_precision - 0.3)

    # Fallback 케이스: expected_keywords가 답변에 있으면 faithfulness=1.0
    if case.get("force_fallback") and \
       any(k in answer for k in case.get("expected_keywords", [])):
        result.faithfulness = 1.0

    return {
        "id":               case["id"],
        "name":             case["name"],
        "question":         question,
        "category":         state.get("category", "?"),
        "retry_count":      state.get("retry_count", 0),
        "faithfulness":     round(result.faithfulness, 3),
        "answer_relevance": round(result.answer_relevance, 3),
        "context_precision":round(result.context_precision, 3),
        "avg_score":        round((result.faithfulness +
                                   result.answer_relevance +
                                   result.context_precision) / 3, 3),
        "reasoning":        result.reasoning,
        "answer_preview":   answer[:200],
    }

# ─────────────────────────────────────────────────────────────
# 메인 실행
# ─────────────────────────────────────────────────────────────
def run_evaluation(selected_cases: List[int] = None, output_prefix: str = None):
    cases = TEST_CASES if not selected_cases else \
            [c for c in TEST_CASES if c["id"] in selected_cases]

    print(f"\n{'='*64}")
    print(f"  🔬 RAG 자동 평가 시작 | {len(cases)}개 케이스")
    print(f"  시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*64}")

    all_results = []

    for i, case in enumerate(cases, 1):
        print(f"\n[{i}/{len(cases)}] {case['name']}")
        print(f"  질문: '{case['question']}'")

        # 그래프 실행
        force_fallback = case.get("force_fallback", False)
        if force_fallback:
            def always_fail(state):
                return {"is_hallucinated": True,
                        "retry_count": state.get("retry_count", 0) + 1}
            with patch(PATCH_TARGETS[0], side_effect=mock_rag), \
                 patch(PATCH_TARGETS[1], side_effect=mock_rag), \
                 patch(PATCH_TARGETS[2], side_effect=mock_rag), \
                 patch("app.core.security.verifier_node", side_effect=always_fail):
                state: GraphState = {
                    "messages": [HumanMessage(content=case["question"])],
                    "category":"","extracted_model":"","rewritten_query":"",
                    "context":"","generated_answer":"","is_hallucinated":False,"retry_count":0,
                }
                final = app_graph.invoke(state,
                            config={"configurable":{"thread_id":f"eval-{case['id']}"}})
        else:
            final = run_graph(case["question"], f"eval-{case['id']}")

        print(f"  → 분류: {final.get('category')} | retry: {final.get('retry_count',0)}")

        # LLM 평가
        print(f"  → LLM 평가 중...")
        eval_result = evaluate_case(case, final)
        all_results.append(eval_result)

        # 결과 출력
        f  = eval_result["faithfulness"]
        ar = eval_result["answer_relevance"]
        cp = eval_result["context_precision"]
        avg = eval_result["avg_score"]

        def score_bar(v): return "█" * int(v * 10) + "░" * (10 - int(v * 10))
        print(f"  Faithfulness:      {f:.3f} [{score_bar(f)}]")
        print(f"  Answer Relevance:  {ar:.3f} [{score_bar(ar)}]")
        print(f"  Context Precision: {cp:.3f} [{score_bar(cp)}]")
        print(f"  ─────────────────────────────")
        print(f"  평균 점수:         {avg:.3f} [{score_bar(avg)}]")
        print(f"  근거: {textwrap.fill(eval_result['reasoning'], 60, subsequent_indent='        ')}")

    # 전체 통계
    print(f"\n{'='*64}")
    print(f"  📊 전체 평가 결과 요약")
    print(f"{'='*64}")
    print(f"  {'케이스명':<35} {'충실도':>6} {'관련성':>6} {'정밀도':>6} {'평균':>6}")
    print(f"  {'-'*63}")
    total_avg = 0
    for r in all_results:
        print(f"  {r['name']:<35} "
              f"{r['faithfulness']:>6.3f} "
              f"{r['answer_relevance']:>6.3f} "
              f"{r['context_precision']:>6.3f} "
              f"{r['avg_score']:>6.3f}")
        total_avg += r["avg_score"]
    print(f"  {'-'*63}")
    total_avg /= len(all_results)
    print(f"  {'전체 평균':>35} {'':>6} {'':>6} {'':>6} {total_avg:>6.3f}")
    grade = "🟢 우수" if total_avg >= 0.8 else "🟡 보통" if total_avg >= 0.6 else "🔴 개선 필요"
    print(f"\n  종합 판정: {grade} (평균 {total_avg:.3f})")

    # JSON 저장
    if output_prefix:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = PROJECT_ROOT / f"{output_prefix}_{ts}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"timestamp": ts, "total_avg": total_avg,
                       "results": all_results}, f, ensure_ascii=False, indent=2)
        print(f"\n  📄 결과 저장: {out_path}")

    return all_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG LLM-as-a-Judge 자동 평가")
    parser.add_argument("--case", type=int, nargs="+",
                        help="평가할 케이스 ID (예: --case 1 3)")
    parser.add_argument("--output", type=str, default=None,
                        help="결과 JSON 저장 파일 접두사 (예: --output report)")
    args = parser.parse_args()
    run_evaluation(selected_cases=args.case, output_prefix=args.output)
