from src.agent.client import InstructorClient
from src.agent.dsl import LiuClawDecision
from src.agent.prompts import SYSTEM_PROMPT, generate_user_prompt
from typing import Dict, Any

class LiuClawAgent:
    """
    The main agent class that coordinates the reasoning loop using vLLM on H200.
    """
    def __init__(self, base_url: str = "http://localhost:8000/v1", api_key: str = "liuclaw-local-key"):
        self.client = InstructorClient(base_url=base_url, api_key=api_key)

    def decide_mutations(self, 
                         kan_state: Dict[str, Any], 
                         pipeline_health: Dict[str, Any],
                         model_name: str = "Qwen/Qwen2.5-32B-Instruct") -> LiuClawDecision:
        """
        Sends the state to vLLM (e.g., Qwen 2.5) and receives a guaranteed parsed Pydantic object.
        """
        # 1. Generate user prompt with context payload
        user_prompt = generate_user_prompt(
            kan_state_dict=kan_state,
            pipeline_health_dict=pipeline_health
        )

        # 2. Call local vLLM server via instructor (JSON mode)
        decision = self.client.get_structured_response(
            response_model=LiuClawDecision,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            model_name=model_name
        )

        return decision
