# WELD·BOT v3.2 (All-in-One Arc Flash Edition)
---
# 1. 팀 소개
  
   ## 팀명 
    
   ## 팀원 소개 
|이름|역할|GitHub|
|------|---|---|
|김도영|팀장|[<img src="https://img.shields.io/badge/github-black?style=for-the-badge&logo=GitHub&logoColor=white">](https://github.com/rubyheartsping)|
|김민정|팀원|[<img src="https://img.shields.io/badge/github-black?style=for-the-badge&logo=GitHub&logoColor=white">](https://github.com/minjeong-kim-dev)|
|신승훈|팀원|[<img src="https://img.shields.io/badge/github-black?style=for-the-badge&logo=GitHub&logoColor=white">](https://github.com/seunghun92-lab)|
|송주엽|팀원|[<img src="https://img.shields.io/badge/github-black?style=for-the-badge&logo=GitHub&logoColor=white">](https://github.com/JUYEOP024)|
|정희영|팀원|[<img src="https://img.shields.io/badge/github-black?style=for-the-badge&logo=GitHub&logoColor=white">](https://github.com/JUNGHEEYOUNG9090)
    
# 2. 개발 배경

로봇 용접 매뉴얼은 복잡한 공정 조건과 다양한 장비 설정을 포함하고 있어, 숙련 작업자가 아니면 이해하기 어렵다는 한계가 있다.
또한 매뉴얼 관리가 비효율적으로 이루어질 경우, 품질 저하 및 작업 오류로 이어질 수 있다.

본 개발은 LLM 기반 문서 분석 및 생성 기술을 적용하여, 로봇 용접 매뉴얼을 자동으로 생성·요약하고, 상황별 가이드를 제공하여 제조업에 뛰어드는 지원자, 신입은 물론 숙련자들에게도 도움을 줄 수 있는 시스템을 구현하는 것을 목표로 한다.  
# 3. 기술 스택 및 사용 모델
## 기술스택
| 영역 | 스택 |
|---|---|
| Language | Python 3.12 |
| Backend API | FastAPI, Uvicorn, Pydantic |
| Frontend | Streamlit, Requests |
| LLM Orchestration | LangChain, LangGraph |
| LLM/Embedding | OpenAI Chat Models (gpt-5.x/gpt-4o), text-embedding-3-small |
| Retrieval | PostgreSQL (pgvector), BM25(rank-bm25), EnsembleRetriever |
| Reranker | BAAI/bge-reranker-v2-m3, sentence-transformers, torch |
| Auth | JWT(python-jose), OAuth(Authlib), Session/Cookie |
| Data Ingestion | marker, pypdf, pymupdf4llm |
| Infra | AWS S3(boto3), SSH Tunnel(sshtunnel/paramiko), Poetry |

## 사용 모델
- `model_fast`: 기본 `gpt-5.2` (재작성/분류/검증)
- `model_accurate`: 기본 `gpt-5.2` (최종 답변 생성)
- `evaluation_model`: 기본 `gpt-4o` (LLM-as-a-Judge)
- Embedding: `text-embedding-3-small`
- Reranker: `BAAI/bge-reranker-v2-m3` (로컬 파일 기반)

# 4. 아키텍쳐 및 플로우차트

## 시스템 아키텍쳐
```mermaid
graph TD
    User([User Browser]) <--> FE[Streamlit UI]
    
    subgraph "Integrated Engine (FastAPI)"
        FE <--> API_Chat[SSE /chat]
        API_Chat --> Graph[LangGraph Workflow]
        Graph --> Reranker[Singleton GPU Reranker]
        Reranker --> Hybrid[Hybrid Search: BM25 + PGVector]
    end
    
    subgraph "Persistence & Intelligence"
        Graph --> History[(AWS RDS History)]
        Graph --> Judge[LLM-as-a-Judge / Chat Logs]
    end
```

## Branch 구조
```
main
 └──develop
      ├── heartsping     # 김도영 작업 브랜치
      ├── kmj            # 김민정 작업 브랜치
      ├── sjy            # 송주엽 작업 브랜치
      ├── ssh            # 신승훈 작업 브랜치
      └── jhy            # 정희영 작업 브랜치
```

## AWS 배포 설계

```
[사용자 브라우저]
       │
       ▼
[Application Load Balancer]
       │
       ▼
[EC2 / ECS] ── FastAPI (app/main.py)
       │              │
       │         [LangGraph 워크플로우]
       │              │
       │    ┌─────────┴──────────┐
       │    ▼                   ▼
       │ [OpenAI API]    [S3 RAW_DATA]
       │                        │
       │                   ingest_all.py
       │                        │
       │                        ▼
       └───────── [RDS PostgreSQL(pgvector) + S3]
       
```

### 폴더 및 파일 구조
```
WELD-BOT v4.0 Project
├── .env                        # 로컬 환경 변수
├── README.md                   # 프로젝트 문서
├── pyproject.toml
├── poetry.lock                 # Poetry 의존성 잠금 파일
├── run_all.py                  # 통합 실행 엔트리 (Tunnel + Backend + Frontend)
├── run_all.bat / run_all.sh    # OS별 실행 스크립트
├── run_tunnel.py               # SSH 터널 백그라운드 프로세스
├── main.py                     # FastAPI 백엔드 엔트리포인트
├── v4_app.py                   # Streamlit 프론트엔드 엔트리포인트
├── kill_all.bat / kill_all.sh  # 실행 프로세스 종료 스크립트
├── check_db_indexes.py         # DB 인덱스 점검 유틸
├── check_dims.py               # 벡터 차원 점검 유틸
├── cleanup_s3.py / list_s3.py  # S3 정리/조회 유틸
├── marker.py                   # 문서 파싱/마크다운 변환 스크립트
├── app/                        # 백엔드 핵심 모듈
│   ├── main.py
│   ├── app_logic.py
│   ├── api/                    # auth/chat/admin 라우터
│   ├── agents/                 # LangGraph 워크플로우/전문가 에이전트
│   ├── core/                   # 설정/보안/히스토리/프롬프트
│   ├── infrastructure/aws/     # Bedrock/OpenSearch/S3/RDS 연동
│   ├── ingest/                 # 적재/전처리 파이프라인
│   ├── rag/                    # retriever/reranker/pipeline
│   ├── schemas/                # 상태/스키마 정의
│   ├── services/               # 서비스 계층
│   └── vectorstore/            # pgvector 스토어 관리
├── frontend/                   # Streamlit UI
│   ├── main_page.py / main_page_logout.py
│   ├── login.py / signup.py / auth_ui.py
│   ├── chat.py / chat_ui.py
│   ├── admin_ui.py / monitoring_ui.py
│   └── image/
│       ├── streamlit1.jpg
│       └── image.png
├── scripts/                    # 운영/유지보수 스크립트
│   ├── check_rds_persistence.py
│   ├── create_hnsw_index.py
│   ├── ingest_all.py
│   └── init_admin_settings.py
├── data/                       # 런타임/캐시/처리 결과 데이터
│   ├── runtime_model_config.json
│   ├── cache/
│   ├── processed/uploads_md/
│   └── vector_cache/
├── domain/                     # 도메인별 문서 루트
│   ├── electrical/docs/
│   ├── robotics/docs/
│   └── welding/docs/
├── models/                     # 로컬 모델 파일 (BGE reranker 등)
├── infrastructure/aws/s3_utils.py
└── logs/
```

## Data Flow
```
사용자 질문 (Streamlit)
    │
    ▼
/chat (SSE) → stream_chat_response()
    │
    ├─ thread_id 기준 대화 이력 로드 후 최근 3턴(6개 메시지) 유지
    ▼
① rewriter_node
    - 현장 은어(JARGON) 정규화 + 브랜드/에러코드 보강 + LLM 쿼리 확장
    - routing_hint 생성: SOCIAL / GENERAL / TECHNICAL
    │
    ├─ SOCIAL  → social_node (RAG 우회, 즉시 응답) → END
    └─ 그 외    → supervisor_node
                   │
                   ▼
② supervisor_node (robotics / welding / electrical / general 분류)
    │
    ├─ general → general_node (RAG 우회) → END
    └─ robotics|welding|electrical
           │
           ▼
③ specialist_node
    - run_rag_pipeline
      → Hybrid Retriever(PGVector + BM25)
      → Cross-Encoder Reranker(Threshold 0.5, Top 4)
      → 도메인 답변 생성
           │
           ▼
④ Guard & Verify
    - check_domain_mismatch: 불일치 시 supervisor_reroute (최대 1회), 초과 시 fallback
    - verifier_node: 환각 검증
      - 실패 & retry_count < 2  → feedback_rewriter → 같은 specialist 재실행
      - 실패 & retry_count >= 2 → fallback
      - 통과                     → END
           │
           ▼
⑤ 응답/로그
    - 최종 generated_answer만 SSE 청크로 전송
    - AsyncPostgresSaver로 이력 저장
    - LLM-as-a-Judge 비동기 평가 후 chat_logs 기록
```

### 제약 조건

| 항목 | 설정값 | 의미 |
|---|---|---|
| `retry_count` | 최대 **2** | 3번째 실패 → fallback |
| `routing_retry` | 최대 **1** | 도메인 재분류 1회 이후 → fallback |
| 대화 이력 | 실행 시 최근 **3턴(6개 메시지)** 유지 | 장기 대화 이력 폭증 방지 |
| Zero-hit 처리 | Context 없음/짧음(<30자) | 재작성 루프 또는 verifier 경로로 복구 |
| Retriever 가중치 | 기본 0.6/0.4 (Vector/BM25), 기술질의 시 BM25 0.7 | 에러코드/모델명 질의 정밀도 강화 |

# 5.기능

## RAG
### Advanced RAG 파이프라인

```
검색 쿼리
    ↓
Hybrid Retriever (EnsembleRetriever)
    ├─ PGVector Search        — 의미 기반 (가중치 0.6)
    └─ BM25 Keyword Search    — 에러코드 정확 매칭 (가중치 0.4)
    ↓
Cross-Encoder Reranker
    모델: BAAI/bge-reranker-v2-m3
    threshold: 0.5 미만 제거
    top_n: 4개 최종 반환
    ↓
Context 문자열 (출처 메타데이터 포함)
```

### RAG 테스트
![rag테스트이미지](./rag_test.png)

## 1. 회원가입 & 로그인

## 2. ChatBot

## 3. 관리자페이지
### &nbsp;&nbsp;✅PDF 추가, 삭제
### &nbsp;&nbsp;✅모델변경
### &nbsp;&nbsp;✅유저리스트

## 4. DataBase 
### &nbsp;&nbsp;✅PostgresSQL
### &nbsp;&nbsp;✅pgvector
### &nbsp;&nbsp;✅S3

  
# 5. WBS
  
# 6. ERD
  
# 7. 시연
  
# 8. 소감
  |이름|회고|
|------|---|
|김도영|갓도영|
|김민정|2|
|신승훈|3|
|송주엽|4|
|정희영|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|
# 9. 참고자료
## 기술스택
