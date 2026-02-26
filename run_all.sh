#!/bin/bash
# run_all.sh - WELD·BOT v4.0 통합 실행 스크립트 (Mac/Linux)

GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 WELD·BOT v4.0 하이브리드 아키텍처 가동 시작...${NC}"

echo -e "1️⃣ SSH 터널링 백그라운드 구동 중..."
# 백그라운드에서 터널을 항시 유지하여, 이후 DB 연결 시 지연을 없앰
poetry run python run_tunnel.py > tunnel.log 2>&1 &
sleep 3 # 터널 연결될 때까지 대기

echo -e "${GREEN}2️⃣ 서버사이드 (Auth + RAG 챗봇 API) 백엔드 시작 중...${NC}"
# 통합본인 main:app을 실행 (auth_router 포함)
poetry run uvicorn main:app --host 0.0.0.0 --port 8000 > backend.log 2>&1 &
BACKEND_PID=$!
sleep 2

echo -e "${GREEN}3️⃣ Streamlit 프론트엔드 UI 시작 중...${NC}"
# 프론트엔드 포어그라운드 시각화
poetry run streamlit run v4_app.py --server.port 8501