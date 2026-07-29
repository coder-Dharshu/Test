"""
Greencare AI — Master Launcher (No Docker, No Redis Required)
=============================================================
Starts all 4 Lego services in separate background processes:
  Lego 2 → http://localhost:8001  (CPU Triage)
  Lego 3 → http://localhost:8002  (VLM Vision Engine)
  Lego 1 → http://localhost:8000  (API Gateway + Serialization)
  Lego 4 → http://localhost:8501  (HITL Dashboard — Streamlit)

Usage:
  python run_all.py
"""

import os
import sys
import time
import signal
import threading
import subprocess
import webbrowser
import urllib.request
import urllib.error

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Load .env ──────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(BASE, ".env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    print("❌  GROQ_API_KEY not set. Please add it to .env")
    sys.exit(1)

PYTHON = sys.executable

# ── Service definitions ────────────────────────────────────────────────────
SERVICES = [
    {
        "name"   : "Lego2-CPU-Triage",
        "cmd"    : [PYTHON, "-m", "uvicorn", "lego2_triage.triage_service:app",
                    "--host", "0.0.0.0", "--port", "8001"],
        "port"   : 8001,
        "health" : "http://localhost:8001/health",
        "color"  : "\033[94m",
    },
    {
        "name"   : "Lego3-VLM-Engine",
        "cmd"    : [PYTHON, "-m", "uvicorn", "lego3_groq.groq_engine:app",
                    "--host", "0.0.0.0", "--port", "8002"],
        "port"   : 8002,
        "health" : "http://localhost:8002/health",
        "color"  : "\033[95m",
    },
    {
        "name"   : "Lego1-API-Gateway",
        "cmd"    : [PYTHON, "-m", "uvicorn", "lego1_gateway.main_standalone:app",
                    "--host", "0.0.0.0", "--port", "8000"],
        "port"   : 8000,
        "health" : "http://localhost:8000/health",
        "color"  : "\033[92m",
    },
    {
        "name"   : "Smart-Triage-HITL-Dashboard",
        "cmd"    : [PYTHON, "-m", "streamlit", "run",
                    "smart_triage/smart_triage_hitl_app.py",
                    "--server.port", "8501",
                    "--server.address", "0.0.0.0",
                    "--server.headless", "true"],
        "port"   : 8501,
        "health" : "http://localhost:8501/healthz",
        "color"  : "\033[93m",
    },
]

RESET = "\033[0m"
BOLD  = "\033[1m"
RED   = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"

processes: list[subprocess.Popen] = []
restart_counts: dict[int, int] = {}       # service_index → restart count
MAX_RESTARTS = 3
RESTART_COOLDOWN_SEC = 4


# ── Port management (Windows) ─────────────────────────────────────────────

def kill_port_holder(port: int) -> bool:
    """
    Kill any process currently bound to the given TCP port on Windows.
    Uses netstat to find the PID and taskkill to terminate it.
    Returns True if a process was killed, False if port was already free.
    """
    if sys.platform != "win32":
        return False

    try:
        result = subprocess.run(
            ["netstat", "-ano", "-p", "TCP"],
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 5 and f":{port}" in parts[1] and parts[3] == "LISTENING":
                pid = parts[4]
                if pid and pid != "0":
                    print(f"  [cleanup] Port {port} held by PID {pid} — killing...")
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", pid],
                        capture_output=True, timeout=5,
                    )
                    time.sleep(1)
                    return True
    except Exception as exc:
        print(f"  [cleanup] Warning: port check failed for {port}: {exc}")
    return False


def ensure_port_free(port: int, max_wait: int = 5) -> bool:
    """Verify a port is free, waiting up to max_wait seconds after cleanup."""
    kill_port_holder(port)
    deadline = time.time() + max_wait
    while time.time() < deadline:
        try:
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("0.0.0.0", port))
                return True
        except OSError:
            time.sleep(0.5)
    return False


# ── Service lifecycle ──────────────────────────────────────────────────────

def wait_for(url: str, timeout: int = 90) -> bool:
    """Poll a health endpoint until it responds or timeout expires."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=3)
            return True
        except Exception:
            time.sleep(1.5)
    return False


def stream_output(proc: subprocess.Popen, name: str, color: str):
    """Stream a child process's stdout/stderr to the console with a tag prefix."""
    try:
        for line in proc.stdout:
            sys.stdout.write(f"{color}[{name}]{RESET} {line}")
            sys.stdout.flush()
    except (ValueError, OSError):
        pass  # stdout closed during shutdown


def launch_service(svc: dict) -> subprocess.Popen:
    """Spawn a service subprocess and start streaming its output."""
    env = os.environ.copy()
    proc = subprocess.Popen(
        svc["cmd"],
        cwd=BASE,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    threading.Thread(
        target=stream_output,
        args=(proc, svc["name"], svc["color"]),
        daemon=True,
    ).start()
    return proc


def shutdown_all():
    """Terminate all child processes cleanly."""
    print(f"\n{BOLD}[STOP] Shutting down all services...{RESET}")
    for proc in processes:
        try:
            if proc.poll() is None:
                if sys.platform == "win32":
                    # Use taskkill /T to kill the entire process tree
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                        capture_output=True, timeout=5,
                    )
                else:
                    proc.terminate()
        except Exception:
            pass

    # Wait briefly for all processes to exit
    for proc in processes:
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
        except Exception:
            pass

    print("All services stopped. Goodbye!")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print(f"\n{BOLD}[*] Greencare AI -- Starting All Services{RESET}")
    print(f"{'=' * 55}")

    # Create required directories
    for d in ["temp_uploads", "pending_review", "final_database",
              "rejected", "lego2_temp", "extracted_assets", "exports"]:
        os.makedirs(os.path.join(BASE, d), exist_ok=True)

    # ── Phase 1: Clean up any zombie processes holding our ports ──────────
    print(f"\n{BOLD}[*] Checking for port conflicts...{RESET}")
    all_ports = [svc["port"] for svc in SERVICES]
    for port in all_ports:
        if not ensure_port_free(port):
            print(f"{RED}[!!] Port {port} is still in use after cleanup — "
                  f"manually kill the process or wait.{RESET}")
            sys.exit(1)
    print(f"{GREEN}[OK] All ports free.{RESET}")

    # ── Phase 2: Start Lego 2 & 3 first (gateway depends on them) ────────
    for svc in SERVICES[:2]:
        print(f"\n[..] Starting {svc['name']}...")
        proc = launch_service(svc)
        processes.append(proc)
        if wait_for(svc["health"]):
            print(f"{GREEN}[OK] {svc['name']} ready.{RESET}")
        else:
            print(f"{YELLOW}[!!] {svc['name']} didn't respond in time — "
                  f"continuing anyway.{RESET}")

    # ── Phase 3: Start Gateway and HITL ───────────────────────────────────
    for svc in SERVICES[2:]:
        print(f"\n[..] Starting {svc['name']}...")
        proc = launch_service(svc)
        processes.append(proc)
        time.sleep(3)

    print(f"\n{BOLD}{'=' * 55}{RESET}")
    print(f"{BOLD}[LIVE] Greencare AI — Smart Triage is running!{RESET}")
    print(f"  [1] API Gateway      --> http://localhost:8000/docs")
    print(f"  [2] CPU Triage       --> http://localhost:8001/docs")
    print(f"  [3] VLM Engine       --> http://localhost:8002/docs")
    print(f"  [4] HITL Dashboard   --> http://localhost:8501  (Streamlit)")
    print(f"{BOLD}{'=' * 55}{RESET}")
    print("\nPress Ctrl+C to stop all services.\n")

    # Auto-open dashboards after a delay
    def open_browser():
        time.sleep(12)
        webbrowser.open("http://localhost:8000/docs")
        time.sleep(3)
        webbrowser.open("http://localhost:8501")
    threading.Thread(target=open_browser, daemon=True).start()

    # ── Phase 4: Monitor and restart crashed services ─────────────────────
    try:
        while True:
            time.sleep(2)
            for i, (svc, proc) in enumerate(zip(SERVICES, processes)):
                if proc.poll() is not None:
                    count = restart_counts.get(i, 0)
                    if count >= MAX_RESTARTS:
                        print(f"\n{RED}[FATAL] {svc['name']} crashed {MAX_RESTARTS} "
                              f"times — giving up.{RESET}")
                        continue

                    print(f"\n{YELLOW}[!!] {svc['name']} exited "
                          f"(code {proc.returncode}) — "
                          f"restarting ({count+1}/{MAX_RESTARTS})...{RESET}")

                    # Cooldown: let the OS release the socket
                    time.sleep(RESTART_COOLDOWN_SEC)
                    ensure_port_free(svc["port"])

                    new_proc = launch_service(svc)
                    processes[i] = new_proc
                    restart_counts[i] = count + 1

    except KeyboardInterrupt:
        shutdown_all()


if __name__ == "__main__":
    main()
