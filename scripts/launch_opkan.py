# scripts/launch_opkan.py
import subprocess
import time
import os
import sys
import signal

def launch():
    """
    Master Launcher for OpKAN:
    1. Launches live_session.py in the background (publishes data).
    2. Launches src/ui/tui/app.py in the foreground (visualizes data).
    """
    project_root = os.getcwd()
    env = os.environ.copy()
    env["PYTHONPATH"] = project_root
    
    # Ensure telemetry file is fresh
    tel_path = os.path.join(project_root, "data/telemetry.json")
    if os.path.exists(tel_path):
        os.remove(tel_path)

    print("🚀 Starting OpKAN Math Engine (Background)...")
    train_proc = subprocess.Popen(
        [sys.executable, "scripts/live_session.py"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    print("📺 Starting Terminal Telemetry (Foreground)...")
    try:
        # Start the TUI
        tui_proc = subprocess.run(
            [sys.executable, "src/ui/tui/app.py"],
            env=env
        )
    except KeyboardInterrupt:
        print("\n🛑 Shutdown requested.")
    finally:
        print("🧹 Cleaning up background processes...")
        train_proc.terminate()
        try:
            train_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            train_proc.kill()
        print("✅ OpKAN Session Terminated.")

if __name__ == "__main__":
    launch()
