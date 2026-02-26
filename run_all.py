import subprocess
import time
import sys
import os
from pathlib import Path

def cleanup_ports():
    """Kills lingering processes on known ports before startup."""
    print("🧹 Cleaning up old processes (Ports 8000, 8501, 8502, 15432)...")
    try:
        # Kill by process name
        subprocess.run(["pkill", "-f", "uvicorn"], stderr=subprocess.DEVNULL)
        subprocess.run(["pkill", "-f", "streamlit"], stderr=subprocess.DEVNULL)
        subprocess.run(["pkill", "-f", "run_tunnel.py"], stderr=subprocess.DEVNULL)
        subprocess.run(["pkill", "-f", "python3 run_all.py"], stderr=subprocess.DEVNULL)
        
        # Kill by port
        for port in [8000, 8501, 8502, 15432]:
            try:
                result = subprocess.run(["lsof", "-t", f"-i:{port}"], capture_output=True, text=True)
                pids = result.stdout.strip().split()
                for pid in pids:
                    if pid:
                        subprocess.run(["kill", "-9", pid], stderr=subprocess.DEVNULL)
            except Exception:
                pass
        time.sleep(2)  # Give ports time to be released globally
    except Exception as e:
        print(f"⚠️ Cleanup error: {e}")

def run_all():
    """
    [V3.1] Master Launcher
    Starts both FastAPI Backend and Streamlit Frontend concurrently.
    """
    root_dir = Path(__file__).resolve().parent
    
    print("\n" + "="*60)
    print("🚀 Industrial RAG All-in-One Master Launcher (v3.1)")
    print("="*60 + "\n")

    cleanup_ports()

    # 0. Start SSH Tunnel Helper (run_tunnel.py)
    print("🚇 Starting SSH Tunnel (run_tunnel.py)...")
    tunnel_proc = subprocess.Popen(
        ["poetry", "run", "python", "run_tunnel.py"],
        cwd=str(root_dir)
    )
    time.sleep(2) # Give tunnel time to establish

    # 1. Start FastAPI Backend (main.py)
    print("\n📡 Starting FastAPI Backend (main.py)...")
    backend_proc = subprocess.Popen(
        ["poetry", "run", "python", "main.py"],
        cwd=str(root_dir)
    )

    # Wait for backend to initialize (BM25 loading can take a long time)
    print("⏳ Waiting for backend to fully load BM25 and start accepting connections...")
    import urllib.request
    from urllib.error import URLError
    
    max_wait = 180  # wait up to 3 minutes
    start_wait = time.time()
    while True:
        try:
            # Try to fetch the OpenAPI docs as a ping
            urllib.request.urlopen("http://127.0.0.1:8000/docs", timeout=2)
            print("✅ Backend is ready!")
            break
        except (URLError, ConnectionError):
            if time.time() - start_wait > max_wait:
                print("❌ Backend failed to start within time limit.")
                sys.exit(1)
            time.sleep(2)

    # 2. Start Streamlit Frontend (v4_app.py)
    print("\n🖥️ Starting Streamlit Frontend (v4_app.py)...")
    frontend_proc = subprocess.Popen(
        ["poetry", "run", "streamlit", "run", "v4_app.py"],
        cwd=str(root_dir)
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
        tunnel_proc.terminate()
        backend_proc.terminate()
        frontend_proc.terminate()
        print("Done.")

if __name__ == "__main__":
    run_all()
