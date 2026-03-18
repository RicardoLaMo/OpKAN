# scripts/launch_opkan.py
import subprocess
import time
import os
import sys

def check_dependencies():
    """Verify that all required TUI and Engine dependencies are installed."""
    required = ["textual", "rich", "textual_plotext", "torch", "pandas", "numpy"]
    missing = []
    for lib in required:
        try:
            # Map dash to underscore for import
            mod_name = lib.replace("-", "_")
            __import__(mod_name)
        except ImportError:
            missing.append(lib)
    if missing:
        print(f"❌ Missing dependencies: {', '.join(missing)}")
        print("💡 Run: pip install textual rich textual-plotext torch pandas numpy")
        return False
    return True

def launch():
    """
    Master Launcher for OpKAN:
    1. Launches live_session.py in the background (publishes data).
    2. Launches src/ui/tui/app.py in the foreground (visualizes data).
    """
    if not check_dependencies():
        return

    project_root = os.getcwd()
    env = os.environ.copy()
    env["PYTHONPATH"] = project_root
    
    # Ensure data directory exists
    os.makedirs(os.path.join(project_root, "data"), exist_ok=True)
    
    # Ensure telemetry file is fresh
    tel_path = os.path.join(project_root, "data/telemetry.json")
    if os.path.exists(tel_path):
        os.remove(tel_path)

    log_path = os.path.join(project_root, "data/engine.log")
    print(f"🚀 Starting OpKAN Math Engine (Logging to {log_path})...")
    
    # Open log file for the background process
    with open(log_path, "w") as log_file:
        train_proc = subprocess.Popen(
            [sys.executable, "scripts/live_session.py"],
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        # Wait a moment for the engine to initialize and create the telemetry file
        print(f"⏳ Waiting for engine initialization (PID: {train_proc.pid})...")
        max_wait = 30 # Increased timeout for H200 data load
        waited = 0
        while not os.path.exists(tel_path) and waited < max_wait:
            time.sleep(1)
            waited += 1
            if waited % 5 == 0:
                print(f"  ... still waiting ({waited}/{max_wait}s) ...")
            if train_proc.poll() is not None:
                print("❌ Engine crashed immediately. Last 10 lines of logs:")
                os.system(f"tail -n 10 {log_path}")
                return

        if not os.path.exists(tel_path):
            print("❌ Telemetry file not created. Engine may be stuck loading data.")
            print("💡 Check data/engine.log for details.")
            train_proc.terminate()
            return

        print("✅ Engine Ready. Launching Terminal Telemetry...")
        time.sleep(1) # Final settle time
        try:
            # Start the TUI
            subprocess.run(
                [sys.executable, "src/ui/tui/app.py"],
                env=env,
                check=True
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
