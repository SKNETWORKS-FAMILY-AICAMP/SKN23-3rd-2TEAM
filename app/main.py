# ============================================================
# [FastAPI 애플리케이션 메인] app/main.py
# ============================================================
# 기동 방법: uvicorn app.main:app --reload --port 8000
# ============================================================
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 FastAPI Server is waking up...")
    yield
    print("🛑 FastAPI Server is shutting down gracefully...")
    print("💤 Shutdown complete.")

app = FastAPI(
    title="산업용 기술지원 챗봇 API",
    description="로봇/용접/전기 분야 현장 매뉴얼 기반 RAG 챗봇 (LangGraph + GPT-4o)",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS: Streamlit 프론트엔드 연동용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 운영 시 특정 도메인으로 제한하세요
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────
# [라우터 등록] app/api/routes/chat.py
# ──────────────────────────────────────────────
app.include_router(chat.router)

# ──────────────────────────────────────────────
# [엔드포인트 3] 헬스 체크
# ──────────────────────────────────────────────
@app.get("/health", summary="서버 상태 확인")
async def health_check():
    return {"status": "ok", "service": "industrial-chatbot-api"}


# ──────────────────────────────────────────────
# [로컬 실행 진입점]
# ──────────────────────────────────────────────
# 터미널에서 직접 실행: python app/main.py
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
