from src.agent.client import InstructorClient
from src.agent.dsl import StrategicDecision, ReflexDecision
from src.agent.prompts import (
    SYSTEM_1_PROMPT, SYSTEM_2_PROMPT,
    generate_system_1_user_prompt, generate_system_2_user_prompt
)
from typing import Dict, Any

class LiuClawAgent:
    """
    The main agent class that implements Dual-Process (Fast and Slow) thinking.
    """
    def __init__(self, base_url: str = "http://localhost:8000/v1", api_key: str = "liuclaw-local-key"):
        self.client = InstructorClient(base_url=base_url, api_key=api_key)

    def think_fast(self, step: int, edge_stats: dict, loss_delta: float) -> ReflexDecision:
        """System 1: Reflexive maintenance."""
        user_prompt = generate_system_1_user_prompt(step, edge_stats, loss_delta)
        return self.client.get_structured_response(
            response_model=ReflexDecision,
            system_prompt=SYSTEM_1_PROMPT,
            user_prompt=user_prompt,
            model_name="Qwen/Qwen2.5-7B-Instruct" # Use smaller model for speed if possible
        )

    def think_slow(self, history: dict, regime_data: dict, model_state: dict) -> StrategicDecision:
        """System 2: Strategic structural review."""
        user_prompt = generate_system_2_user_prompt(history, regime_data, model_state)
        return self.client.get_structured_response(
            response_model=StrategicDecision,
            system_prompt=SYSTEM_2_PROMPT,
            user_prompt=user_prompt,
            model_name="Qwen/Qwen2.5-32B-Instruct" # Use larger model for reasoning
        )
