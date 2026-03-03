import subprocess
import time
import sys
import os
from pathlib import Path

def cleanup_ports():
    """Kills lingering processes on known ports before startup."""
    print("🧹 Cleaning up old processes (Ports 8000, 8501, 8502, 15432)...")
    try:
        if os.name == "nt":
            target_ports = {8000, 8501, 8502, 15432}
            netstat = subprocess.run(
                ["netstat", "-ano"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore",
            )
            killed_pids = set()
            for line in netstat.stdout.splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                if parts[0].upper() not in {"TCP", "UDP"}:
                    continue
                try:
                    local_port = int(parts[1].rsplit(":", 1)[-1])
                except ValueError:
                    continue
                pid = parts[-1]
                if local_port in target_ports and pid.isdigit() and pid not in killed_pids:
                    subprocess.run(["taskkill", "/PID", pid, "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    killed_pids.add(pid)
        else:
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
    # [수정] SSH 터널이 DB와 연결을 완전히 맺을 때까지 충분한 시간 대기
    print("⏳ Waiting for SSH Tunnel to firmly establish (5 seconds)...")
    time.sleep(5)

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
        print("\n🛑 Sending SIGTERM to all workers for graceful shutdown...")
        for p in [tunnel_proc, backend_proc, frontend_proc]:
            if p.poll() is None:
                p.terminate()
        
        # Wait for up to 5 seconds for them to exit cleanly
        for p in [tunnel_proc, backend_proc, frontend_proc]:
            if p.poll() is None:
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print("⚠️ Process did not terminate in time, forcing kill...")
                    p.kill()
                    
        print("✅ All processes terminated cleanly. Exiting.")
if __name__ == "__main__":
    run_all()
