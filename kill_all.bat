@echo off
REM kill_all.bat - WELD·BOT v4.0 통합 종료 및 클린업 스크립트 (Windows)

echo 🛑 WELD·BOT v4.0 서비스를 강제 종료하고 정리합니다...

REM 1. 포트 기준 프로세스 종료
echo ⚡ 포트 8000(FastAPI) 및 8501(Streamlit) 강제 종료 중...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8000') do taskkill /f /pid %%a 2>nul
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8501') do taskkill /f /pid %%a 2>nul

REM 2. 백그라운드 터널 유지 스크립트 특화 종료
echo 🐍 파이썬 숨은 터널 프로세스 종료 중...
wmic process where "name='python.exe' and commandline like '%%run_tunnel.py%%'" call terminate >nul 2>&1

REM 3. 클린업
echo 🗑️ 불필요한 테스트 파일 및 로그 클린업 중...
if exist tunnel.log del /f /q tunnel.log
if exist backend.log del /f /q backend.log
if exist diagnose_db.py del /f /q diagnose_db.py
if exist execute_init_db.py del /f /q execute_init_db.py
if exist init_db.sql del /f /q init_db.sql
if exist tests\start-dev.bat del /f /q tests\start-dev.bat
if exist tests\stop-dev.bat del /f /q tests\stop-dev.bat
if exist tests\login_app.py del /f /q tests\login_app.py

echo ✅ 모든 환경이 깔끔하게 리셋되었습니다!
