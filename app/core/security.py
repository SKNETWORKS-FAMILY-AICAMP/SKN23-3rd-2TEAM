from __future__ import annotations

import os


def aws_region(default: str = "ap-northeast-2") -> str:
    return os.getenv("AWS_REGION", default)


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value
