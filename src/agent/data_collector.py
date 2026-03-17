import json
import os
import torch
import pandas as pd
from datetime import datetime

class LoraDataCollector:
    """
    Collects training data for LoRA fine-tuning of the LiuClaw agent.
    Format: (Context, Decision, LossImprovement)
    """
    def __init__(self, log_path: str = "data/lora_training_data.jsonl"):
        self.log_path = log_path
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def log_trajectory(self, 
                       context: dict, 
                       decision: dict, 
                       pre_loss: float, 
                       post_loss: float):
        """
        Logs a single interaction.
        """
        improvement = pre_loss - post_loss
        
        entry = {
            "timestamp": datetime.now().isoformat(),
            "context": context,
            "decision": decision,
            "reward": improvement,
            "success": improvement > 0
        }
        
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

# Global collector instance
collector = LoraDataCollector()
