import pytest
from unittest.mock import MagicMock
from src.agent.core import LiuClawAgent
from src.agent.dsl import LiuClawDecision, EdgeMutation, RegimeThesis

def test_liuclaw_decision_validation_v2():
    # Validates the new Master Payload schema
    data = {
        "training_command": "CONTINUE",
        "reasoning": "Reasoning for the decision.",
        "mutations": [
            {
                "edge_id": "L0_N0_to_L1_N1",
                "action": "REPLACE",
                "formula": "torch.exp(x)",
                "reasoning": "High vol trend detected."
            },
            {
                "edge_id": "L0_N1_to_L1_N0",
                "action": "PRUNE",
                "reasoning": "Insignificant activation."
            }
        ],
        "regime_analysis": {
            "hmm_transition_detected": True,
            "predicted_regime": 1,
            "thesis_statement": "Volatility expansion due to skew steepening."
        },
        "confidence": 0.95
    }
    decision = LiuClawDecision(**data)
    assert decision.training_command == "CONTINUE"
    assert len(decision.mutations) == 2
    assert decision.regime_analysis.predicted_regime == 1

def test_agent_decide_mutations_vllm_mock():
    # Mock the InstructorClient to return the new LiuClawDecision schema
    agent = LiuClawAgent()
    agent.client.get_structured_response = MagicMock(return_value=LiuClawDecision(
        training_command="CONTINUE",
        reasoning="Global reasoning",
        mutations=[EdgeMutation(
            edge_id="L0_N0_to_L1_N1",
            action="REPLACE",
            formula="torch.pow(x, 2)",
            reasoning="Quadratic fit for S."
        )],
        regime_analysis=RegimeThesis(
            hmm_transition_detected=False
        ),
        confidence=1.0
    ))
    
    decision = agent.decide_mutations(
        kan_state={"L0_N0_to_L1_N1": {"mean": 1.2}},
        pipeline_health={"bid_ask_spread": 0.01}
    )
    
    assert decision.training_command == "CONTINUE"
    assert decision.mutations[0].edge_id == "L0_N0_to_L1_N1"
    assert decision.mutations[0].action == "REPLACE"
    agent.client.get_structured_response.assert_called_once()
