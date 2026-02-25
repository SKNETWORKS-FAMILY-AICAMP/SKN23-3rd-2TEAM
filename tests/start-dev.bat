@echo off
start "FASTAPI" cmd /k c:\Users\Playdata\multimodal\multi_venv\Scripts\python.exe -m uvicorn auth_api:app --app-dir c:\Users\Playdata\test --reload --reload-dir c:\Users\Playdata\test --port 8000
start "STREAMLIT" cmd /k c:\Users\Playdata\multimodal\multi_venv\Scripts\python.exe -m streamlit run c:\Users\Playdata\test\login_app.py
