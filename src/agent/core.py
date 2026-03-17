from src.agent.client import InstructorClient
from src.agent.dsl import LiuClawDecision, ReflexDecision, StrategicDecision
from src.agent.prompts import SYSTEM_PROMPT, generate_user_prompt
from typing import Dict, Any

class LiuClawAgent:
    """
    The main agent class that coordinates the reasoning loop using vLLM on H200.

    Implements a dual-process (System 1 / System 2) interface:
      - think_fast()  -- reflex path, called by _system_1_worker
      - think_slow()  -- strategic path, called by _system_2_worker
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
        decision = self.client.get_structured_response(
            response_model=LiuClawDecision,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            model_name=model_name
        )
        return decision

    def think_fast(self,
                   step: int,
                   edge_stats: Dict[str, Any],
                   loss_delta: float) -> ReflexDecision:
        """
        System 1 (reflex) reasoning: fast heuristic decisions for edge pruning
        and learning-rate adjustment.  Override or mock in tests.
        """
        raise NotImplementedError("think_fast must be implemented or mocked.")

    def think_slow(self,
                   history: Dict[str, Any],
                   regime_data: Dict[str, Any],
                   model_state: Dict[str, Any]) -> StrategicDecision:
        """
        System 2 (strategic) reasoning: deliberate topological mutations and
        regime analysis via the LLM backend.  Override or mock in tests.
        """
        raise NotImplementedError("think_slow must be implemented or mocked.")
