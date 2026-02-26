#!/bin/bash
# kill_all.sh - WELD·BOT v4.0 통합 종료 및 클린업 스크립트 (Mac/Linux)

RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}🛑 WELD·BOT v4.0 서비스를 강제 종료하고 정리합니다...${NC}"

# 1. 포트 기준 프로세스 종료 (TCP 8000, 8501)
echo "⚡ 포트 8000(FastAPI) 및 8501(Streamlit) 강제 종료 중..."
lsof -ti :8000 | xargs kill -9 2>/dev/null
lsof -ti :8501 | xargs kill -9 2>/dev/null

# 2. 백그라운드 터널 유지 프로세스 종료
echo "🐍 파이썬 숨은 터널 프로세스 종료 중..."
pkill -f run_tunnel.py 2>/dev/null

# 3. 찌꺼기 파일 클린업 (보안 및 디스크 관리)
echo -e "${RED}🗑️ 불필요한 테스트 파일 및 로그 클린업 중...${NC}"
rm -f .backend.pid .frontend.pid tunnel.log backend.log
rm -f diagnose_db.py execute_init_db.py init_db.sql
rm -f tests/start-dev.bat tests/stop-dev.bat tests/login_app.py

echo -e "${RED}✅ 모든 환경이 깔끔하게 리셋되었습니다!${NC}"
