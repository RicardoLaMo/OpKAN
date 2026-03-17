import time
import pytest
from unittest.mock import MagicMock
from src.engine.coordinator import EngineCoordinator
from src.agent.core import LiuClawAgent
from src.agent.dsl import LiuClawDecision, EdgeMutation, RegimeThesis
from src.engine.queues import context_queue, decision_queue

class MockModel:
    def __init__(self, layers: list):
        self.layers = layers

def test_async_vllm_mutation_processing():
    # 1. Setup mock agent
    agent = MagicMock(spec=LiuClawAgent)
    
    def slow_reasoning(*args):
        time.sleep(0.1) # vLLM is fast!
        return LiuClawDecision(
            training_command="CONTINUE",
            reasoning="Market trend is quadratic.",
            mutations=[EdgeMutation(
                edge_id="L0_N0_to_L1_N0",
                action="REPLACE",
                formula="torch.pow(x, 2)",
                reasoning="Quadratic trend"
            )],
            regime_analysis=RegimeThesis(hmm_transition_detected=False),
            confidence=0.9
        )
    
    agent.decide_mutations.side_effect = slow_reasoning
    
    # 2. Setup mock model
    mock_layer = MagicMock()
    mock_model = MockModel(layers=[mock_layer])
    
    coordinator = EngineCoordinator(agent)
    coordinator.start_agent_thread()
    
    # Ensure queues are empty
    while not decision_queue.empty(): decision_queue.get()
    
    # 3. Simulate training loop iterations
    start_time = time.time()
    iterations = 0
    
    while time.time() - start_time < 0.5:
        iterations += 1
        
        if iterations == 5:
            coordinator.request_mutation({"L0_N0_to_L1_N0": {"mean": 1.0}}, {"health": "good"})
            
        status = coordinator.apply_pending_mutations(mock_model)
        assert status == "CONTINUE"
        
        time.sleep(0.01)
        
    coordinator.stop_agent_thread()
    
    # 4. Assertions
    # mock_layer.swap_edge should be called via mutate_edge
    # parse_edge_id for 'L0_N0_to_L1_N0' should give layer_idx=0, in_idx=0, out_idx=0
    mock_layer.swap_edge.assert_called_once()
