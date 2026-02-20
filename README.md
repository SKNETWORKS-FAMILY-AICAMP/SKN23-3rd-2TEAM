# SKN23-3rd-2TEAM

## Project Structure

| 경로 | 파일/폴더 | 역할 |
|---|---|---|
| `app/agents/` | `supervisor.py` | 사용자 질문을 분석해 전문 에이전트로 라우팅하는 오케스트레이터 |
| `app/agents/specialists/` | `welding.py`, `robotics.py`, `safety.py`, `electrical.py` | 분야별(용접/로봇/안전/전기) 특화 응답 생성 |
| `app/rag/` | `embeddings.py`, `retriever.py`, `reranker.py`, `pipeline.py` | 임베딩 생성, 문서 검색, 재정렬, RAG 파이프라인 실행 |
| `app/vectorstore/` | `base.py`, `chroma_store.py`, `pgvector_store.py` | 벡터 DB 추상화 및 구현체(Chroma, PGVector) |
| `app/ingest/` | `pipeline.py`, `chunking.py`, `loaders/*` | 문서 로딩, 청크 분할, 인덱싱 전처리 파이프라인 |
| `app/api/routes/` | `chat.py`, `health.py` | 챗봇 API 엔드포인트 및 헬스체크 라우트 |
| `app/infrastructure/aws/` | `s3_client.py`, `bedrock_client.py`, `opensearch_client.py`, `rds_pgvector_client.py` | AWS 서비스 연동 클라이언트 계층 |
| `app/core/` | `config.py`, `logging.py`, `prompts.py`, `security.py` | 공통 설정/로깅/프롬프트/보안 유틸 |
| `domain/<분야>/docs/` | 분야별 문서 디렉터리 | 로컬 샘플/캐시 문서 저장소 (원본은 S3 권장) |
| `configs/` | `settings.yaml`, `agents.yaml`, `rag.yaml` | 시스템 설정, 에이전트 정책, RAG 파라미터 관리 |
| `infra/terraform/` | `README.md` | Terraform 기반 인프라 관리 영역 |
| `infra/cdk/` | `README.md` | AWS CDK 기반 인프라 관리 영역 |
| `scripts/` | `ingest_all.py`, `build_index.py`, `eval_retrieval.py` | 데이터 적재, 인덱스 구축, 검색 성능 평가 자동화 |
| `tests/` | `unit/`, `integration/` | 유닛 테스트와 통합 테스트 골격 |

## AWS Deployment Note

- 문서 원본 저장: Amazon S3
- 벡터 검색: Amazon OpenSearch Serverless 또는 Amazon RDS PostgreSQL + pgvector
- 모델 추론: Amazon Bedrock
- 비밀/자격 증명: IAM Role + AWS Secrets Manager/SSM Parameter Store
