import time
import pytest
from unittest.mock import MagicMock
from src.engine.coordinator import EngineCoordinator
from src.agent.core import LiuClawAgent
from src.agent.dsl import StrategicDecision, ReflexDecision, EdgeMutation, RegimeThesis
from src.engine.queues import reflex_queue, strategic_queue, reflex_decision_queue, strategic_decision_queue


class MockModel:
    def __init__(self, layers):
        self.layers = layers


def _drain_queues():
    for q in (reflex_queue, strategic_queue, reflex_decision_queue, strategic_decision_queue):
        while not q.empty():
            try:
                q.get_nowait()
                q.task_done()
            except Exception:
                pass


def test_reflex_decision_processing():
    """System 1 (fast) path: prune edges and adjust LR."""
    _drain_queues()

    agent = MagicMock(spec=LiuClawAgent)
    agent.think_fast.return_value = ReflexDecision(
        reasoning="Edge L0_N0_to_L1_N0 has near-zero activation, pruning.",
        prunes=["L0_N0_to_L1_N0"],
        lr_adjustment=0.9,
    )

    mock_layer = MagicMock()
    mock_model = MockModel(layers=[mock_layer])

    coordinator = EngineCoordinator(agent)
    coordinator.start_threads()

    coordinator.request_reflex(step=10, edge_stats={"L0_N0_to_L1_N0": {"l1": 0.001}}, loss_delta=-0.5)
    time.sleep(0.3)  # Allow system_1_worker to process

    coordinator.apply_pending_mutations(mock_model)
    coordinator.stop_threads()

    # The pruned edge should trigger swap_edge on the mock layer
    mock_layer.swap_edge.assert_called_once()
    _drain_queues()


def test_strategic_decision_processing():
    """System 2 (slow) path: REPLACE mutation on a specific edge."""
    _drain_queues()

    agent = MagicMock(spec=LiuClawAgent)
    agent.think_slow.return_value = StrategicDecision(
        reasoning="Quadratic basis fits vol-of-vol curvature better.",
        mutations=[EdgeMutation(
            edge_id="L0_N0_to_L1_N0",
            action="REPLACE",
            formula="torch.pow(x, 2)",
            reasoning="Quadratic trend detected.",
        )],
        regime_analysis=RegimeThesis(hmm_transition_detected=False),
        training_command="CONTINUE",
    )

    mock_layer = MagicMock()
    mock_model = MockModel(layers=[mock_layer])

    coordinator = EngineCoordinator(agent)
    coordinator.start_threads()

    coordinator.request_strategic(
        history={"losses": [0.5, 0.4, 0.3]},
        regime_data={"regime": 0},
        model_state={"layer_0": "summary"},
    )
    time.sleep(0.3)

    coordinator.apply_pending_mutations(mock_model)
    coordinator.stop_threads()

    mock_layer.swap_edge.assert_called_once()
    _drain_queues()
