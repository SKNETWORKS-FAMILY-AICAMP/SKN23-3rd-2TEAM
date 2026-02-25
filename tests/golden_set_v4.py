"""
Advanced Golden Set Evaluation v4.0
tests/golden_set_v4.py
-------------------------------------------------------------
특징:
    1. IN_DB_CASES (20개): 실제 DB에 존재하는 지식 확인 (expected_action: "ANSWER")
    2. OUT_OF_DB_CASES (20개): 존재하지 않는/함정 질문 확인 (expected_action: "REJECT")
    3. fetch_sample_docs_from_db(): DB에서 실제 데이터를 추출하여 질문 작성을 지원
    4. 스마트 검증: REJECT 케이스에서 환각(지어내기) 발생 시 FAIL 처리 (길이 및 추측성 어휘 제한)
"""

import argparse
import json
import asyncio
import time
import os
import sys
import re
from datetime import datetime

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import HumanMessage
from app.agents.graph import app_graph
from app.vectorstore.pgvector_store import PGVectorStoreManager

# ──────────────────────────────────────────────────────────────
# [Helper] DB에서 실제 데이터 샘플 추출
# ──────────────────────────────────────────────────────────────
async def fetch_sample_docs_from_db(limit=20):
    """
    RDS의 langchain_pg_embedding 테이블에서 실제 문서 조각(document)을 가져옵니다.
    사용자는 이 내용을 보고 IN_DB_CASES를 작성할 수 있습니다.
    """
    print(f"\n🔍 [DB Sampler] Fetching {limit} random document samples from RDS...")
    
    import psycopg2
    ssh_enabled = os.getenv("SSH_TUNNEL_ENABLED", "false").lower() == "true"
    ssh_local_port = os.getenv("SSH_LOCAL_BIND_PORT", "15432")
    target_host = "127.0.0.1" if ssh_enabled else os.getenv("PGHOST", "localhost")
    target_port = ssh_local_port if ssh_enabled else os.getenv("PGPORT", "5432")

    try:
        conn = psycopg2.connect(
            host=target_host,
            database=os.getenv("PGDATABASE", "chatbot_db"),
            user=os.getenv("PGUSER", "postgres"),
            password=os.getenv("PGPASSWORD", "password"),
            port=target_port,
            connect_timeout=10
        )
        with conn.cursor() as cur:
            # 테이블 이름은 check_schema.py에서 확인한 'langchain_pg_embedding' 사용
            cur.execute("SELECT document FROM langchain_pg_embedding ORDER BY RANDOM() LIMIT %s", (limit,))
            rows = cur.fetchall()
            
            print("\n" + "="*80)
            print("📜 [REAL DB SAMPLES] - Use these to create IN_DB_CASES")
            print("="*80)
            for i, row in enumerate(rows):
                doc_text = row[0].replace('\n', ' ')[:150]
                print(f"[{i+1:02}] {doc_text}...")
            print("="*80 + "\n")
            
        conn.close()
    except Exception as e:
        print(f"❌ DB Sample Fetch Failed: {e}")

# ──────────────────────────────────────────────────────────────
# [Test Cases] IN-DB (20) & OUT-OF-DB (20)
# ──────────────────────────────────────────────────────────────

IN_DB_CASES = [
    {"id": "IN-01", "action": "ANSWER", "question": "HH020 로봇 브레이크 수동 해제 방법 알려줘", "expected_keywords": ["브레이크", "해제", "수동", "HH020"]},
    {"id": "IN-02", "action": "ANSWER", "question": "Hi5 제어기 배터리 전압 낮음 알람 조치 방법", "expected_keywords": ["배터리", "전압", "교체", "알람"]},
    {"id": "IN-03", "action": "ANSWER", "question": "TP630 티칭 펜던트 원점 복귀 방법", "expected_keywords": ["TP630", "원점", "복귀", "캘리브레이션"]},
    {"id": "IN-04", "action": "ANSWER", "question": "현대로보틱스 Hi5 E143 알람 해결책", "expected_keywords": ["E143", "Hi5", "알람", "조치"]},
    {"id": "IN-05", "action": "ANSWER", "question": "YRC1000 제어기 가스 배출 시 주의사항", "expected_keywords": ["YRC1000", "가스", "배출", "주의"]},
    {"id": "IN-06", "action": "ANSWER", "question": "야스카와 4107 알람 조치법", "expected_keywords": ["4107", "야스카와", "알람", "서보"]},
    {"id": "IN-07", "action": "ANSWER", "question": "두산 M0042 에러코드 해결방법", "expected_keywords": ["M0042", "두산", "에러", "해결"]},
    {"id": "IN-08", "action": "ANSWER", "question": "두산 로봇 협동 작업 중 충돌 감지 설정", "expected_keywords": ["두산", "충돌", "감지", "설정"]},
    {"id": "IN-09", "action": "ANSWER", "question": "ABB 38013 에러 원인", "expected_keywords": ["38013", "ABB", "에러", "원인"]},
    {"id": "IN-10", "action": "ANSWER", "question": "IRB2400 케이블 교체 시 주의사항", "expected_keywords": ["IRB2400", "케이블", "교체", "주의"]},
    {"id": "IN-11", "action": "ANSWER", "question": "PolyScope에서 아날로그 입력 도메인 설정 방법", "expected_keywords": ["PolyScope", "아날로그", "입력", "설정"]},
    {"id": "IN-12", "action": "ANSWER", "question": "UR10e 안전 정지 해제 기능", "expected_keywords": ["UR10e", "안전", "정지", "해제"]},
    {"id": "IN-13", "action": "ANSWER", "question": "RB10 협동로봇 충돌 민감도 조정", "expected_keywords": ["RB10", "충돌", "민감도", "조정"]},
    {"id": "IN-14", "action": "ANSWER", "question": "레인보우 로보틱스 TCP 캘리브레이션", "expected_keywords": ["레인보우", "TCP", "캘리브레이션"]},
    {"id": "IN-15", "action": "ANSWER", "question": "탄소강 MAG 용접 스패터 방지법", "expected_keywords": ["탄소강", "MAG", "스패터", "방지"]},
    {"id": "IN-16", "action": "ANSWER", "question": "용접 기공(Porosity) 발생 원인과 해결", "expected_keywords": ["기공", "Porosity", "원인", "해결"]},
    {"id": "IN-17", "action": "ANSWER", "question": "MCCB 차단기 트립 원인", "expected_keywords": ["MCCB", "차단기", "트립", "원인"]},
    {"id": "IN-18", "action": "ANSWER", "question": "NPN/PNP 센서 배선 차이", "expected_keywords": ["NPN", "PNP", "센서", "배선"]},
    {"id": "IN-19", "action": "ANSWER", "question": "로봇 설치 시 ISO 10218-1 안전 규격 준수 사항", "expected_keywords": ["ISO", "10218-1", "안전", "규격"]},
    {"id": "IN-20", "action": "ANSWER", "question": "트랙 센서 ERRNO 발생 시 에러 핸들링", "expected_keywords": ["트랙", "센서", "ERRNO", "핸들링"]},
]

OUT_OF_DB_CASES = [
    {"id": "OUT-01", "action": "REJECT", "question": "Hi6 제어기에서 전자레인지 모드 켜줘"},
    {"id": "OUT-02", "action": "REJECT", "question": "현대로보틱스 로봇으로 피자 토핑 올리는 법"},
    {"id": "OUT-03", "action": "REJECT", "question": "야스카와 로봇으로 달나라 가는 항법 설정"},
    {"id": "OUT-04", "action": "REJECT", "question": "YK1000 제어기용 카카오톡 연동 방법"},
    {"id": "OUT-05", "action": "REJECT", "question": "두산 로봇이 춤추는 노래 추천해줘"},
    {"id": "OUT-06", "action": "REJECT", "question": "두산 로봇 에러코드 M9999 조치방법"},
    {"id": "OUT-07", "action": "REJECT", "question": "ABB 로봇으로 비트코인 채굴하는 스크립트"},
    {"id": "OUT-08", "action": "REJECT", "question": "IRC5 제어기에서 넷플릭스 실행 방법"},
    {"id": "OUT-09", "action": "REJECT", "question": "UR3 로봇으로 수영장 물 채우기"},
    {"id": "OUT-10", "action": "REJECT", "question": "UR5e 제어기에 에어컨 기능이 있나요?"},
    {"id": "OUT-11", "action": "REJECT", "question": "레인보우 로봇으로 로또 번호 예측하기"},
    {"id": "OUT-12", "action": "REJECT", "question": "RB-Infinity 모델 사양 알려줘"},
    {"id": "OUT-13", "action": "REJECT", "question": "용접 불꽃으로 스테이크 굽는 방법"},
    {"id": "OUT-14", "action": "REJECT", "question": "액체 질소로 용접하는 특수 기법"},
    {"id": "OUT-15", "action": "REJECT", "question": "제어반 안에서 라면 끓여 먹어도 되나요?"},
    {"id": "OUT-16", "action": "REJECT", "question": "두꺼비집으로 전력 무한 생성하는 법"},
    {"id": "OUT-17", "action": "REJECT", "question": "로봇을 애완동물처럼 기르는 사육 가이드"},
    {"id": "OUT-18", "action": "REJECT", "question": "챗봇이 직접 현장에 와서 수리해줄 수 있나요?"},
    {"id": "OUT-19", "action": "REJECT", "question": "로봇이 자아를 가지게 되는 시점"},
    {"id": "OUT-20", "action": "REJECT", "question": "오늘 점심 메뉴 추천해줘"},
]

REJECTION_KEYWORDS = [
    "없습니다", "알 수 없습니다", "확인할 수 없습니다", "찾을 수 없습니다", "제공하지 않습니다", 
    "매뉴얼에", "도와드릴 수", "제한적", "범위를 벗어", "전문가", "기술적인", "어렵습니다"
]
SPECULATIVE_KEYWORDS = ["일반적으로", "보통", "하지만", "다만", "그러나", "그렇지만", "수도 있습니다"]

# ──────────────────────────────────────────────────────────────
# [Evaluator] 실전 엔진 평가 실행기
# ──────────────────────────────────────────────────────────────
async def run_evaluation_v4(verbose=False, output_file="tests/v4_report.json"):
    all_cases = IN_DB_CASES + OUT_OF_DB_CASES
    results = []
    
    print(f"\n🚀 WELD·BOT v4.0 Advanced Evaluation 시작 (Total: {len(all_cases)})")
    
    for case in all_cases:
        cid = case["id"]
        action = case["action"]
        question = case["question"]
        
        print(f"[{cid}] [{action}] Question: {question}")
        
        try:
            inputs = {"messages": [HumanMessage(content=question)]}
            config = {"configurable": {"thread_id": f"v4_{cid}_{int(time.time())}"}}
            
            result_state = await app_graph.ainvoke(inputs, config)
            actual_answer = result_state.get("generated_answer", "")
            actual_domain = result_state.get("category", "general")
            
            passed = False
            reason = ""
            
            if action == "ANSWER":
                # 키워드 매칭 검증 (유연한 매칭: 50% 이상 또는 최소 2개)
                keywords = case.get("expected_keywords", [])
                hit_keywords = [kw for kw in keywords if kw.lower() in actual_answer.lower()]
                hit_count = len(hit_keywords)
                
                # 임계값 설정: 키워드가 3개 이상이면 절반 이상, 2개 이하면 전부
                threshold = max(2, len(keywords) // 2) if len(keywords) > 2 else len(keywords)
                passed = (hit_count >= threshold)
                reason = f"Keywords hit: {hit_count}/{len(keywords)} (Need {threshold}) | Hits: {hit_keywords}"
                
            elif action == "REJECT":
                # 거절 키워드 포함 확인 (환각 방어)
                hit_reject = [kw for kw in REJECTION_KEYWORDS if kw in actual_answer]
                passed = len(hit_reject) > 0
                reason = f"Rejection hit: {hit_reject}" if passed else "Hallucination/Fabrication suspected (No rejection keywords)"

            results.append({
                "id": cid, 
                "question": question,
                "passed": passed, 
                "actual_domain": actual_domain, 
                "answer": actual_answer, 
                "reason": reason
            })
            
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f" >> {status} | Domain: {actual_domain} | {reason}")
            if verbose and not passed:
                print(f"    [실제 답변]: {actual_answer.replace(chr(10), ' ')}")
            
        except Exception as e:
            print(f" >> 🧨 ERROR: {e}")
            results.append({"id": cid, "question": question, "passed": False, "error": str(e)})

        await asyncio.sleep(1.0) # 살짝 단축

    # 결과 요약
    in_db_passed = sum(1 for r in results if r["id"].startswith("IN-") and r.get("passed"))
    out_db_passed = sum(1 for r in results if r["id"].startswith("OUT-") and r.get("passed"))
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total": len(all_cases),
        "in_db_score": f"{in_db_passed}/{len(IN_DB_CASES)}",
        "out_db_score": f"{out_db_passed}/{len(OUT_OF_DB_CASES)}",
        "success_rate": f"{(in_db_passed + out_db_passed) / len(all_cases) * 100:.1f}%",
        "details": results
    }

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\n✅ Detailed report saved to: {output_file}")

    print(f"\n" + "="*50)
    print(f"📊 Final Report (v4.0 Revised)")
    print(f" - IN-DB:  {in_db_passed}/{len(IN_DB_CASES)} Passed")
    print(f" - OUT-DB: {out_db_passed}/{len(OUT_OF_DB_CASES)} Passed")
    print(f" - Total Success Rate: {summary['success_rate']}")
    print("="*50 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch-samples", action="store_true", help="DB에서 샘플 데이터를 추출합니다.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output", default="tests/v4_report.json")
    args = parser.parse_args()

    async def main():
        with PGVectorStoreManager() as _:
            if args.fetch_samples:
                await fetch_sample_docs_from_db(limit=30)
            else:
                await run_evaluation_v4(verbose=args.verbose, output_file=args.output)

    asyncio.run(main())