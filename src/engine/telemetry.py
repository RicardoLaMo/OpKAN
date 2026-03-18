import json
import os
import time
import fcntl
from typing import Dict, Any, List

class TelemetryStore:
    """
    A file-based telemetry store with flock-based locking to prevent
    data loss during concurrent access from training and TUI processes.
    """
    def __init__(self, path: str = "data/telemetry.json"):
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._initialize_store()

    def _initialize_store(self):
        if not os.path.exists(self.path):
            initial_data = {
                "step": 0,
                "pde_loss": 0.0,
                "option_price": 0.0,
                "delta": 0.5,
                "gamma": 0.0,
                "vega": 0.0,
                "throughput": 0,
                "regime": "INITIALIZING",
                "logs": [],
                "s1_active": False,
                "s2_active": False,
                "dual_mode": True,
                "active": False
            }
            self.write(initial_data)

    def write(self, data: Dict[str, Any]):
        """Writes data to the telemetry file with an exclusive lock."""
        try:
            with open(self.path, "w") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                json.dump(data, f)
                fcntl.flock(f, fcntl.LOCK_UN)
        except Exception as e:
            print(f"Telemetry Write Error: {e}")

    def read(self) -> Dict[str, Any]:
        """Reads data from the telemetry file with a shared lock and retry logic."""
        for _ in range(5):
            try:
                if not os.path.exists(self.path):
                    return {}
                with open(self.path, "r") as f:
                    fcntl.flock(f, fcntl.LOCK_SH)
                    content = f.read()
                    fcntl.flock(f, fcntl.LOCK_UN)
                    if not content: return {}
                    return json.loads(content)
            except (json.JSONDecodeError, OSError):
                time.sleep(0.02)
        return {}

    def log_event(self, message: str):
        """
        Thread-safe log append. Reads, modifies, and writes with an exclusive lock
        to ensure no logs are lost during concurrent main-loop writes.
        """
        from datetime import datetime
        try:
            # Re-open with r+ for read-modify-write under a single lock
            with open(self.path, "r+") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                content = f.read()
                if not content:
                    data = {}
                else:
                    data = json.loads(content)
                
                logs = data.get("logs", [])
                logs.append({
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "message": message
                })
                data["logs"] = logs[-30:] # Increased buffer
                
                f.seek(0)
                f.truncate()
                json.dump(data, f)
                fcntl.flock(f, fcntl.LOCK_UN)
        except Exception as e:
            print(f"Telemetry Log Error: {e}")

# Global singleton
telemetry = TelemetryStore()
