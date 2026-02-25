# WELD·BOT v3.2 (All-in-One Arc Flash Edition)

본 프로젝트는 현장 기술 문서를 기반으로 정밀한 용접 및 로봇 기술 지원을 제공하는 Multi-Agent RAG 시스템입니다. v3.2 아키텍처는 루트 기반의 통합 구조와 하드웨어 가속 싱글톤 리랭커를 탑재하고 있습니다.

## 🏗️ System Architecture (v3.2)

```mermaid
graph TD
    User([User Browser]) <--> FE[Streamlit UI]
    
    subgraph "Integrated Engine (FastAPI)"
        FE <--> API_Chat[SSE /chat]
        API_Chat --> Graph[LangGraph Workflow]
        Graph --> Reranker[Singleton GPU Reranker]
        Reranker --> Hybrid[Hybrid Search: BM25 + pgvector]
    end
    
    subgraph "Persistence & Intelligence"
        Graph --> History[(AWS RDS History)]
        Graph --> Web[Tavily Search]
    end
```

## 📂 Project Structure (v3.2)

| 경로 | 역할 |
|---|---|
| [main.py](file:///Users/jy/3rd-2TEAM/SKN23-3rd-2TEAM/main.py) | **Engine Entry**. FastAPI 서버, 데이터 로드, 하위 기능 통합 관리 |
| [run_all.py](file:///Users/jy/3rd-2TEAM/SKN23-3rd-2TEAM/run_all.py) | **Master Launcher**. 백엔드와 프론트엔드를 동시에 가동하는 메인 스크립트 |
| `app/agents/` | LangGraph 멀티 에이전트 워크플로우 및 노드 로직 |
| `app/rag/` | 하이브리드 검색 및 싱글톤 GPU 리랭커 시스템 |
| `frontend/` | WELD·BOT 전용 다크 테마 UI 및 관리자 대시보드 |
| `data/` | BM25 캐시, 현장 은어 사전(Jargon), 정제된 매뉴얼 데이터 |

---

## 🚀 Execution Guide

1. **Environment**: 프로젝트 루트의 `.env` 파일에 API 키 및 RDS 정보를 정확히 입력하세요.
2. **One-Command Start**:
   ```bash
   python run_all.py
   ```
3. **Details**: 상세한 파일별 기능 및 유지보수 가이드는 [TEAM_HANDOVER_GUIDE.md](file:///Users/jy/3rd-2TEAM/SKN23-3rd-2TEAM/TEAM_HANDOVER_GUIDE.md)를 참조하세요.
