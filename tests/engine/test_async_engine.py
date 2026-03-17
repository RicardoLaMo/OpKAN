import time
import pytest
from unittest.mock import MagicMock
from src.engine.coordinator import EngineCoordinator
from src.agent.core import LiuClawAgent
from src.agent.dsl import LiuClawDecision, EdgeMutation
from src.engine.queues import context_queue, decision_queue

class MockModel:
    def __init__(self, layers: list):
        self.layers = layers

def test_async_mutation_processing():
    # 1. Setup mock agent that is "slow"
    agent = MagicMock(spec=LiuClawAgent)
    
    def slow_reasoning(*args):
        time.sleep(0.5) # Simulate LLM thinking
        return LiuClawDecision(
            reasoning="Market trend is quadratic.",
            mutations=[EdgeMutation(
                layer_idx=0, input_idx=0, output_idx=0,
                symbolic_expression="torch.pow(x, 2)",
                explanation="Quadratic trend"
            )],
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
    mutation_requested = False
    
    # Run loop for 1 second
    while time.time() - start_time < 1.0:
        iterations += 1
        
        # At iteration 10, request mutation
        if iterations == 10:
            coordinator.request_mutation({"dummy": "stats"}, "Low Vol", "Flat surface")
            mutation_requested = True
            
        # Try to apply mutations (should be empty initially)
        coordinator.apply_pending_mutations(mock_model)
        
        # Busy loop to simulate training math
        time.sleep(0.01)
        
    coordinator.stop_agent_thread()
    
    # 4. Assertions
    # Training loop should have run many iterations (>50) despite the slow agent
    assert iterations > 50, f"Training loop blocked! Only {iterations} iterations."
    
    # The mutation should have been applied
    mock_layer.swap_edge.assert_called_once()
    
def test_context_queue_overwriting():
    # Verify that the context queue doesn't grow indefinitely
    # (maxsize=1 as defined in queues.py)
    agent = MagicMock(spec=LiuClawAgent)
    coordinator = EngineCoordinator(agent)
    
    # Push 3 contexts without starting worker
    coordinator.request_mutation({"id": 1}, "R", "V")
    coordinator.request_mutation({"id": 2}, "R", "V")
    coordinator.request_mutation({"id": 3}, "R", "V")
    
    assert context_queue.qsize() == 1
    # Check that it kept the most recent one (id=3)
    # The way we implemented it (get_nowait and put), it should be id=3
    last_context = context_queue.get()
    assert last_context[0]["id"] == 3
