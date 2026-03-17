import pytest
from unittest.mock import MagicMock
from src.agent.core import LiuClawAgent
from src.agent.dsl import LiuClawDecision, EdgeMutation

def test_liuclaw_decision_validation():
    # Validates that the Pydantic model works as expected
    data = {
        "reasoning": "Market is in high vol regime, observed exponential-like growth in some edges.",
        "mutations": [
            {
                "layer_idx": 0,
                "input_idx": 1,
                "output_idx": 2,
                "symbolic_expression": "torch.exp(x)",
                "explanation": "High volatility requires exponential growth."
            }
        ],
        "confidence": 0.9,
        "regime_adjustment": "Tighten BC constraints for small S."
    }
    decision = LiuClawDecision(**data)
    assert decision.confidence == 0.9
    assert decision.mutations[0].symbolic_expression == "torch.exp(x)"

def test_agent_decide_mutations_mock():
    # Mock the InstructorClient to return a LiuClawDecision
    agent = LiuClawAgent()
    agent.client.get_structured_response = MagicMock(return_value=LiuClawDecision(
        reasoning="Reasoning content",
        mutations=[EdgeMutation(
            layer_idx=0, input_idx=0, output_idx=0, 
            symbolic_expression="torch.pow(x, 2)", 
            explanation="Quadratic trend detected."
        )],
        confidence=0.8
    ))
    
    decision = agent.decide_mutations(
        kan_stats={"edge_0_0_0": {"mean": 1.2, "std": 0.5}},
        current_regime="Regime 1: High Vol",
        vol_info="Surface is flat"
    )
    
    assert decision.confidence == 0.8
    assert decision.mutations[0].symbolic_expression == "torch.pow(x, 2)"
    agent.client.get_structured_response.assert_called_once()
