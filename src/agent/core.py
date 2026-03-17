from src.agent.client import InstructorClient
from src.agent.dsl import LiuClawDecision
from src.agent.prompts import SYSTEM_PROMPT, generate_user_prompt
from typing import Dict, Any

class LiuClawAgent:
    """
    The main agent class that coordinates the reasoning loop for KAN mutations.
    """
    def __init__(self, base_url: str = "http://localhost:8000/v1", api_key: str = "empty"):
        self.client = InstructorClient(base_url=base_url, api_key=api_key)

    def decide_mutations(self, 
                         kan_stats: Dict[str, Any], 
                         current_regime: str, 
                         vol_info: str) -> LiuClawDecision:
        """
        Takes current state and returns a structured mutation decision.
        """
        # 1. Format user prompt
        # In a real scenario, kan_stats would be stringified concisely
        user_prompt = generate_user_prompt(
            kan_stats=str(kan_stats),
            current_regime=current_regime,
            vol_surface_info=vol_info
        )

        # 2. Call H200 vLLM server via instructor
        # Note: This is an async call in real life, but client.py uses synchronous instructor for now.
        decision = self.client.get_structured_response(
            response_model=LiuClawDecision,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt
        )

        return decision
