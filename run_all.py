import subprocess
import time
import sys
import os
from pathlib import Path

def run_all():
    """
    [V3.1] Master Launcher
    Starts both FastAPI Backend and Streamlit Frontend concurrently.
    """
    root_dir = Path(__file__).resolve().parent
    
    print("\n" + "="*60)
    print("🚀 Industrial RAG All-in-One Master Launcher (v3.1)")
    print("="*60 + "\n")

    # 1. Start FastAPI Backend (main.py)
    print("📡 Starting FastAPI Backend (main:app)...")
    backend_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"],
        cwd=os.getcwd()
    )

    # Wait for backend to initialize
    time.sleep(3)

    # 2. Start Streamlit Frontend (frontend/main_ui.py)
    print("\n🖥️ Starting Streamlit Frontend (frontend/main_ui.py)...")
    frontend_proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "frontend/main_ui.py"],
        cwd=os.getcwd()
    )

    print("\n" + "✅"*20)
    print("  System is running!")
    print("  - Backend: http://localhost:8000")
    print("  - Frontend: http://localhost:8501")
    print("✅"*20 + "\n")
    print("Press Ctrl+C to stop both services.\n")

    try:
        while True:
            time.sleep(1)
            # Check if processes are still alive
            if backend_proc.poll() is not None:
                print("❌ Backend process terminated.")
                break
            if frontend_proc.poll() is not None:
                print("❌ Frontend process terminated.")
                break
    except KeyboardInterrupt:
        print("\n🛑 Stopping services...")
    finally:
        backend_proc.terminate()
        frontend_proc.terminate()
        print("Done.")

if __name__ == "__main__":
    run_all()
