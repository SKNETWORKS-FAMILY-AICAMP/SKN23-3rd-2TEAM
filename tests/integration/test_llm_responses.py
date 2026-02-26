import sys
import os
from dotenv import load_dotenv

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from app.core.prompts import (
    COMMON_SYSTEM_RULES,
    PRE_CHECK_PROTOCOL,
    SUPERVISOR_PROMPT,
    ROBOTICS_SPECIALIST_PROMPT,
    WELDING_SPECIALIST_PROMPT,
    ELECTRICAL_SPECIALIST_PROMPT,
    GENERAL_PROMPT
)

def build_specialist_prompt(specialist_prompt_template):
    """공통 규칙과 Context 변수가 이미 주입된 완성형 프롬프트 템플릿 반환"""
    # f-string 적용으로 이미 형식에 맞게 문자열이 구성되었으며, {{context}} 가 {context}로 남아있음.
    return ChatPromptTemplate.from_messages([
        ("system", specialist_prompt_template),
        ("human", "{question}")
    ])

def test_specialist_responses_interactive():
    """사용자가 직접 질문과 문맥을 입력하여 각 도메인 전문가(Specialist) LLM의 답변을 테스트합니다."""
    
    print("\n" + "="*60)
    print("🤖 Specialist 대화형 응답(Interactive Response) 테스트")
    print("="*60 + "\n")

    try:
        # GPT-4o-mini 등 사용 가능
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0) 
    except Exception as e:
        print(f"❌ LLM 초기화 실패: {e}")
        return

    # Supervisor Chain
    supervisor_prompt = ChatPromptTemplate.from_messages([
        ("system", SUPERVISOR_PROMPT),
        ("human", "{question}")
    ])
    supervisor_chain = supervisor_prompt | llm

    # General Chain
    general_prompt = ChatPromptTemplate.from_messages([
        ("system", GENERAL_PROMPT),
        ("human", "{question}")
    ])
    general_chain = general_prompt | llm

    # Specialist Chains
    specialist_chains = {
        "ROBOTICS": build_specialist_prompt(ROBOTICS_SPECIALIST_PROMPT) | llm,
        "WELDING": build_specialist_prompt(WELDING_SPECIALIST_PROMPT) | llm,
        "ELECTRICAL": build_specialist_prompt(ELECTRICAL_SPECIALIST_PROMPT) | llm
    }

    while True:
        question = input("\nQ (사용자 질문 입력 / 종료는 q): ").strip()
        if question.lower() == 'q':
            print("테스트를 종료합니다.")
            break
        if not question:
            continue
            
        print("\n⏳ LLM Supervisor가 도메인을 분류 중입니다...")
        try:
            supervisor_res = supervisor_chain.invoke({"question": question})
            domain_decision = supervisor_res.content.strip().upper()
        except Exception as e:
            print(f"❌ API 호출 실패: {e}")
            continue

        # Supervisor 응답 파싱
        matched_domain = None
        for d in ["ROBOTICS", "WELDING", "ELECTRICAL", "GENERAL"]:
            if d in domain_decision:
                matched_domain = d
                break
                
        if not matched_domain:
            print(f"⚠️ Supervisor 분류 실패. 원본 응답: {domain_decision}")
            continue
            
        print(f"✅ Supervisor 분류 결과: [{matched_domain}]")

        # 해당 도메인이 GENERAL인 경우 바로 대답
        if matched_domain == "GENERAL":
            print("\n⏳ GENERAL 담당 LLM이 답변을 생성 중입니다...")
            try:
                res = general_chain.invoke({"question": question})
                print("-" * 60)
                print(f"[AI 답변]\n{res.content}")
                print("-" * 60)
            except Exception as e:
                print(f"❌ API 호출 실패: {e}")
            continue

        # ROBOTICS, WELDING, ELECTRICAL인 경우만 내부 RAG 파이프라인에서 Context를 가져옴
        # (테스트 환경에서는 실제 RAG 모듈을 연동하거나, 고정된 테스트 문맥을 사용)
        print("\n⏳ 사내 매뉴얼(RAG) 검색 중...")
        try:
            from app.rag.pipeline import run_rag_pipeline
            # 테스트 편의성을 위해 첫 번째 키워드를 기반으로 필터링 임시 적용
            filters = {"model_name": "Hi6"} if "hi6" in question.lower() else {}
            context = run_rag_pipeline(question, domain=matched_domain[:5], filters=filters)
            if not context or "(관련 매뉴얼 없음)" in context:
                 context = "(검색 결과 없음)"
                 print("⚠️ 검색된 매뉴얼이 없습니다.")
            else:
                 print("✅ 매뉴얼 검색 완료!")
        except ImportError:
             print("⚠️ RAG 모듈을 찾을 수 없습니다. 빈 문맥(Context)으로 테스트를 진행합니다.")
             context = ""
        except Exception as e:
             print(f"⚠️ RAG 검색 중 오류 발생: {e}. 빈 문맥으로 진행합니다.")
             context = ""

        print(f"\n⏳ LLM Specialist ({matched_domain}) 가 답변을 생성 중입니다...\n")
        try:
            response = specialist_chains[matched_domain].invoke({"question": question, "context": context})
            print("-" * 60)
            print(f"[AI 답변]\n{response.content}")
            print("-" * 60)
        except Exception as e:
            print(f"❌ API 호출 실패: {e}")

if __name__ == "__main__":
    test_specialist_responses_interactive()
