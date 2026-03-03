import os
import sys
import torch
import json
import threading
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
MODEL_CONFIG_PATH = DATA_DIR / "runtime_model_config.json"

SUPPORTED_CHAT_MODELS = [
    "gpt-5.2",
    "gpt-5",
    "gpt-5-mini",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4o",
    "gpt-4o-mini",
]

DEFAULT_MODEL_FAST = os.getenv("MODEL_FAST", "gpt-5.2")      # 쿼리 재작성, 분류, 검증 (Speed)
DEFAULT_MODEL_ACCURATE = os.getenv("MODEL_ACCURATE", "gpt-5.2")  # 실제 답변 생성 (Accuracy)
DEFAULT_EVALUATION_MODEL = os.getenv("EVALUATION_MODEL", "gpt-4o") # LLM-as-a-judge 평가 모델

_MODEL_CONFIG_LOCK = threading.Lock()
_MODEL_CONFIG_CACHE: dict | None = None


def _normalize_model_name(model_name: str) -> str:
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("모델명은 비어 있을 수 없습니다.")
    return model_name.strip()


def _validate_model_name(model_name: str) -> str:
    name = _normalize_model_name(model_name)
    if name not in SUPPORTED_CHAT_MODELS and not name.startswith("gpt-"):
        print(f"⚠️ 권장 모델 목록 외 모델명이 설정되었습니다: {name}")
    return name


def _default_model_settings() -> dict:
    return {
        "model_fast": _validate_model_name(DEFAULT_MODEL_FAST),
        "model_accurate": _validate_model_name(DEFAULT_MODEL_ACCURATE),
        "evaluation_model": _validate_model_name(DEFAULT_EVALUATION_MODEL),
    }


def _read_model_settings_from_disk() -> dict:
    settings = _default_model_settings()
    if not MODEL_CONFIG_PATH.exists():
        return settings

    try:
        payload = json.loads(MODEL_CONFIG_PATH.read_text(encoding="utf-8"))
        raw_fast = payload.get("model_fast", settings["model_fast"])
        raw_accurate = payload.get("model_accurate", settings["model_accurate"])
        raw_eval = payload.get("evaluation_model", settings["evaluation_model"])
        settings["model_fast"] = _validate_model_name(raw_fast)
        settings["model_accurate"] = _validate_model_name(raw_accurate)
        settings["evaluation_model"] = _validate_model_name(raw_eval)
    except Exception as e:
        print(f"⚠️ 모델 설정 파일 로드 실패. 기본값으로 복구합니다: {e}")

    return settings


def _persist_model_settings(settings: dict) -> None:
    MODEL_CONFIG_PATH.write_text(
        json.dumps(settings, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _get_model_settings_cached() -> dict:
    global _MODEL_CONFIG_CACHE
    with _MODEL_CONFIG_LOCK:
        if _MODEL_CONFIG_CACHE is None:
            _MODEL_CONFIG_CACHE = _read_model_settings_from_disk()
        return dict(_MODEL_CONFIG_CACHE)


def get_model_settings() -> dict:
    return _get_model_settings_cached()


def list_available_chat_models() -> list[str]:
    return list(SUPPORTED_CHAT_MODELS)


def set_model_settings(model_fast: str | None = None, model_accurate: str | None = None, evaluation_model: str | None = None) -> dict:
    global _MODEL_CONFIG_CACHE, MODEL_FAST, MODEL_ACCURATE
    with _MODEL_CONFIG_LOCK:
        current = _MODEL_CONFIG_CACHE or _read_model_settings_from_disk()
        next_settings = {
            "model_fast": _validate_model_name(model_fast or current["model_fast"]),
            "model_accurate": _validate_model_name(model_accurate or current["model_accurate"]),
            "evaluation_model": _validate_model_name(evaluation_model or current.get("evaluation_model", DEFAULT_EVALUATION_MODEL)),
        }
        _persist_model_settings(next_settings)
        _MODEL_CONFIG_CACHE = next_settings
        MODEL_FAST = next_settings["model_fast"]
        MODEL_ACCURATE = next_settings["model_accurate"]
        return dict(next_settings)


def get_model_fast() -> str:
    return _get_model_settings_cached()["model_fast"]


def get_model_accurate() -> str:
    return _get_model_settings_cached()["model_accurate"]


# 하위 호환용 전역 상수(기존 import 경로 유지)
MODEL_FAST = get_model_fast()
MODEL_ACCURATE = get_model_accurate()

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
