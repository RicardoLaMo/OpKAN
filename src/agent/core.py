from src.agent.client import InstructorClient
from src.agent.dsl import LiuClawDecision, ReflexDecision, StrategicDecision
from src.agent.prompts import (
    SYSTEM_PROMPT, generate_user_prompt,
    SYSTEM_1_PROMPT, SYSTEM_2_PROMPT,
    generate_system_1_user_prompt, generate_system_2_user_prompt
)
from typing import Dict, Any

class LiuClawAgent:
    """
    The main agent class that implements Dual-Process (Fast and Slow) thinking.
    Coordinates the reasoning loop using vLLM on H200.
    """
    def __init__(self, base_url: str = "http://localhost:8000/v1", api_key: str = "liuclaw-local-key"):
        self.client = InstructorClient(base_url=base_url, api_key=api_key)

    def decide_mutations(self,
                         kan_state: Dict[str, Any],
                         pipeline_health: Dict[str, Any],
                         model_name: str = "Qwen/Qwen2.5-32B-Instruct") -> LiuClawDecision:
        """
        Legacy single-process interface.
        Sends the state to vLLM and receives a guaranteed parsed Pydantic object.
        """
        user_prompt = generate_user_prompt(
            kan_state_dict=kan_state,
            pipeline_health_dict=pipeline_health
        )
        return self.client.get_structured_response(
            response_model=LiuClawDecision,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            model_name=model_name
        )

    def think_fast(self, step: int, edge_stats: Dict[str, Any], loss_delta: float) -> ReflexDecision:
        """System 1: Reflexive maintenance."""
        user_prompt = generate_system_1_user_prompt(step, edge_stats, loss_delta)
        return self.client.get_structured_response(
            response_model=ReflexDecision,
            system_prompt=SYSTEM_1_PROMPT,
            user_prompt=user_prompt,
            model_name="Qwen/Qwen2.5-7B-Instruct"
        )

    def think_slow(self, history: Dict[str, Any], regime_data: Dict[str, Any], model_state: Dict[str, Any]) -> StrategicDecision:
        """System 2: Strategic structural review."""
        user_prompt = generate_system_2_user_prompt(history, regime_data, model_state)
        return self.client.get_structured_response(
            response_model=StrategicDecision,
            system_prompt=SYSTEM_2_PROMPT,
            user_prompt=user_prompt,
            model_name="Qwen/Qwen2.5-32B-Instruct"
        )
