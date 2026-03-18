import json
import os
from typing import Dict, Any, List

class TelemetryStore:
    """
    A lightweight, file-based telemetry store for inter-process communication
    between the PyTorch training loop and the TUI.
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
                "dual_mode": True
            }
            self.write(initial_data)

    def write(self, data: Dict[str, Any]):
        """Writes the current state to the telemetry file."""
        try:
            with open(self.path, "w") as f:
                json.dump(data, f)
        except Exception as e:
            print(f"Telemetry Write Error: {e}")

    def read(self) -> Dict[str, Any]:
        """Reads the current state from the telemetry file."""
        try:
            if not os.path.exists(self.path):
                return {}
            with open(self.path, "r") as f:
                return json.load(f)
        except Exception as e:
            # Return empty dict if file is being written to
            return {}

    def log_event(self, message: str):
        """Appends a log message to the telemetry store."""
        from datetime import datetime
        data = self.read()
        if not data: return
        
        logs = data.get("logs", [])
        logs.append({
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "message": message
        })
        # Keep only last 20 logs
        data["logs"] = logs[-20:]
        self.write(data)

# Global singleton for the process
telemetry = TelemetryStore()
