import os
from pathlib import Path
from huggingface_hub import snapshot_download
from app.core.config import MODELS_DIR, RERANKER_MODEL_NAME, RERANKER_LOCAL_PATH

def get_reranker_model_path():
    """
    Reranker 모델의 로컬 경로를 반환합니다.
    폴더가 없으면 HuggingFace에서 자동으로 다운로드합니다.
    """
    if not RERANKER_LOCAL_PATH.exists():
        print(f"📥 모델이 로컬에 없습니다. 다운로드를 시작합니다: {RERANKER_MODEL_NAME}")
        # HuggingFace에서 모델 다운로드
        snapshot_download(
            repo_id=RERANKER_MODEL_NAME,
            local_dir=RERANKER_LOCAL_PATH,
            local_dir_use_symlinks=False
        )
        print(f"✅ 모델 다운로드 완료: {RERANKER_LOCAL_PATH}")
    else:
        print(f"📦 로컬 모델 파일을 사용합니다: {RERANKER_LOCAL_PATH}")
        
    return str(RERANKER_LOCAL_PATH)

def check_models_ready():
    """
    모든 필수 모델이 준비되었는지 빠르게 확인합니다. (Streamlit 초기화용)
    """
    return RERANKER_LOCAL_PATH.exists()
