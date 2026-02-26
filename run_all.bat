@echo off
REM run_all.bat - WELD·BOT v4.0 통합 실행 스크립트 (Windows)

echo 🚀 WELD·BOT v4.0 하이브리드 아키텍처 가동 시작 (Windows)...

echo 1️⃣ SSH 터널링 백그라운드 구동 중...
REM 백그라운드에서 터널을 항시 유지하여, 이후 DB 연결 시 지연을 없앰
start /B poetry run python run_tunnel.py > tunnel.log 2>&1
timeout /t 3 /nobreak > nul

echo 2️⃣ 서버사이드 (Auth + RAG 챗봇 API) 백엔드 시작 중...
REM 통합본인 main:app을 실행 (auth_router 포함)
start /B poetry run uvicorn main:app --host 0.0.0.0 --port 8000 > backend.log 2>&1
timeout /t 2 /nobreak > nul

echo 3️⃣ Streamlit 프론트엔드 UI 시작 중...
REM 프론트엔드 포어그라운드 시각화
poetry run streamlit run v4_app.py --server.port 8501

echo ==========================================
echo ✅ 실행 완료! 종료하려면 kill_all.bat를 실행하세요.
echo ==========================================
