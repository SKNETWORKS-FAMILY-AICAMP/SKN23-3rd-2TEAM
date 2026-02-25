import os
import sys
import torch
from pathlib import Path
from dotenv import load_dotenv

# 1. 경로 고정 (Cross-Platform)
# 프로젝트 루트 디렉토리 결정 (app/core/config.py의 부모의 부모: SKN23-3rd-2TEAM)
ROOT_DIR = Path(__file__).resolve().parents[2]

# 주요 디렉토리 경로 설정
MODELS_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
LOGS_DIR = ROOT_DIR / "logs"

# 디렉토리 자동 생성
for directory in [MODELS_DIR, DATA_DIR, CACHE_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# 2. 환경 변수 로드 및 검증
load_dotenv(ROOT_DIR / ".env")

def validate_config():
    """
    필수 환경 변수가 있는지 확인하고, 없으면 명확한 가이드를 제공합니다.
    """
    required_vars = ["OPENAI_API_KEY", "PGHOST", "PGUSER", "PGPASSWORD", "PGDATABASE", "TAVILY_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        error_msg = f"""
❌ 필수 환경 변수가 누락되었습니다: {', '.join(missing_vars)}

프로젝트 루트의 `.env` 파일을 확인해 주세요. 
만약 파일이 없다면 `.env.example`을 복사하여 작성해야 합니다.

[가이드]
1. OPENAI_API_KEY: OpenAI API 키 (gpt-4o 사용)
2. PGHOST/USER/PASSWORD/DATABASE: AWS RDS 연결 정보
3. TAVILY_API_KEY: Tavily 웹 검색 API 키
"""
        return False, error_msg
    return True, ""

def get_device():
    """
    최적의 하드웨어 가속 장치를 반환합니다 (CUDA > mps > cpu).
    """
    if torch.cuda.is_available():
        return "cuda"
    elif sys.platform == "darwin" and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

# 3. 모델 설정
MODEL_FAST = "gpt-4o"      # 쿼리 재작성, 분류, 검증 (Speed)
MODEL_ACCURATE = "gpt-4o"      # 실제 답변 생성 (Accuracy)

RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
RERANKER_LOCAL_PATH = MODELS_DIR / "bge-reranker-v2-m3"

EMBEDDING_MODEL_NAME = "text-embedding-3-small"

# 4. 외부 서비스 및 보안
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
ADMIN_SECRET_KEY = os.getenv("ADMIN_SECRET_KEY", "admin1234")

# 5. 리트리버 설정
COLLECTION_NAME = "welding_robotics_manuals"
DEFAULT_VECTOR_WEIGHT = 0.6
DEFAULT_BM25_WEIGHT = 0.4
TECHNICAL_BM25_WEIGHT = 0.7  # 기술 용어/에러코드 포함 시 BM25 가중치 상향
