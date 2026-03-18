"""
Unit tests for RuleBasedFallbackAgent (src/agent/fallback.py).

Covers:
- think_fast: L1 pruning threshold, LR adjustment on rising/stable loss
- think_slow: structural prune mutations, HMM transition detection, no-HMM path
- Output types match the dual-process DSL contracts
"""
import numpy as np
import pytest

from src.agent.fallback import (
    RuleBasedFallbackAgent,
    L1_S1_PRUNE_THRESHOLD,
    L1_S2_PRUNE_THRESHOLD,
    HMM_TRANSITION_MASS,
)
from src.agent.dsl import ReflexDecision, StrategicDecision


@pytest.fixture
def agent():
    return RuleBasedFallbackAgent()


# ── think_fast ───────────────────────────────────────────────────────────────

def test_think_fast_returns_reflex_decision(agent):
    result = agent.think_fast(step=1, edge_stats={}, loss_delta=0.0)
    assert isinstance(result, ReflexDecision)


def test_think_fast_prunes_edges_below_threshold(agent):
    edge_stats = {
        "L0_N0_to_L1_N0": {"l1_norm": L1_S1_PRUNE_THRESHOLD - 0.001, "type": "bspline"},
        "L0_N1_to_L1_N0": {"l1_norm": L1_S1_PRUNE_THRESHOLD + 0.001, "type": "bspline"},
        "L0_N2_to_L1_N0": {"l1_norm": 0.0, "type": "bspline"},
    }
    result = agent.think_fast(step=10, edge_stats=edge_stats, loss_delta=0.0)
    assert "L0_N0_to_L1_N0" in result.prunes
    assert "L0_N2_to_L1_N0" in result.prunes
    assert "L0_N1_to_L1_N0" not in result.prunes


def test_think_fast_no_prunes_when_all_above_threshold(agent):
    edge_stats = {
        f"L0_N{i}_to_L1_N0": {"l1_norm": 0.1, "type": "bspline"} for i in range(5)
    }
    result = agent.think_fast(step=5, edge_stats=edge_stats, loss_delta=0.0)
    assert result.prunes == []


def test_think_fast_decays_lr_on_rising_loss(agent):
    result = agent.think_fast(step=1, edge_stats={}, loss_delta=0.01)
    assert result.lr_adjustment == pytest.approx(0.9)


def test_think_fast_keeps_lr_on_stable_loss(agent):
    result = agent.think_fast(step=1, edge_stats={}, loss_delta=-0.005)
    assert result.lr_adjustment == pytest.approx(1.0)


def test_think_fast_lr_boundary_at_threshold(agent):
    # Exactly at threshold: not > 0.001, so no decay
    result = agent.think_fast(step=1, edge_stats={}, loss_delta=0.001)
    assert result.lr_adjustment == pytest.approx(1.0)


def test_think_fast_skips_non_dict_stats(agent):
    """Malformed entries in edge_stats must not cause a crash."""
    edge_stats = {
        "L0_N0_to_L1_N0": "bad_value",
        "L0_N1_to_L1_N0": {"l1_norm": 0.001, "type": "bspline"},
    }
    result = agent.think_fast(step=1, edge_stats=edge_stats, loss_delta=0.0)
    assert "L0_N0_to_L1_N0" not in result.prunes  # bad entry skipped
    assert "L0_N1_to_L1_N0" in result.prunes       # valid low-L1 entry pruned


# ── think_slow ───────────────────────────────────────────────────────────────

def test_think_slow_returns_strategic_decision(agent):
    result = agent.think_slow(history={}, regime_data={}, model_state={})
    assert isinstance(result, StrategicDecision)


def test_think_slow_prunes_deeply_dormant_bspline(agent):
    model_state = {
        "L0_N0_to_L1_N0": {"l1_norm": L1_S2_PRUNE_THRESHOLD - 0.0001, "type": "bspline"},
        "L0_N1_to_L1_N0": {"l1_norm": L1_S2_PRUNE_THRESHOLD + 0.001,  "type": "bspline"},
    }
    result = agent.think_slow(
        history={"step": 100, "current_regime_id": 0},
        regime_data={},
        model_state=model_state,
    )
    mutation_ids = {m.edge_id for m in result.mutations}
    assert "L0_N0_to_L1_N0" in mutation_ids
    assert "L0_N1_to_L1_N0" not in mutation_ids
    assert all(m.action == "PRUNE" for m in result.mutations)


def test_think_slow_no_mutations_when_all_edges_active(agent):
    model_state = {
        f"L0_N{i}_to_L1_N0": {"l1_norm": 0.1, "type": "bspline"} for i in range(4)
    }
    result = agent.think_slow(
        history={"step": 50}, regime_data={}, model_state=model_state
    )
    assert result.mutations == []


def test_think_slow_ignores_pruned_edge_type(agent):
    """Already-pruned edges (type='pruned') must not generate duplicate mutations."""
    model_state = {
        "L0_N0_to_L1_N0": {"l1_norm": 0.0, "type": "pruned"},
    }
    result = agent.think_slow(
        history={"step": 10}, regime_data={}, model_state=model_state
    )
    assert result.mutations == []


def test_think_slow_detects_regime_transition_high_off_diagonal(agent):
    # Construct a transition matrix where off-diagonal mass > HMM_TRANSITION_MASS
    n = 3
    diag = 1.0 - HMM_TRANSITION_MASS - 0.05  # so off-diag sum is clearly > threshold
    off  = (1.0 - diag) / (n - 1)
    transmat = [[diag if i == j else off for j in range(n)] for i in range(n)]
    regime_data = {"transition_matrix": transmat, "means": [[0], [1], [2]], "covars": [[[1]], [[1]], [[1]]]}
    result = agent.think_slow(
        history={"step": 200, "current_regime_id": 1},
        regime_data=regime_data,
        model_state={},
    )
    assert result.regime_analysis.hmm_transition_detected is True


def test_think_slow_no_transition_on_identity_matrix(agent):
    transmat = [[1.0, 0.0], [0.0, 1.0]]
    regime_data = {"transition_matrix": transmat}
    result = agent.think_slow(
        history={"step": 50, "current_regime_id": 0},
        regime_data=regime_data,
        model_state={},
    )
    assert result.regime_analysis.hmm_transition_detected is False


def test_think_slow_no_transition_without_hmm_data(agent):
    """Before HMM is fitted, regime_data is {}: no transition detected."""
    result = agent.think_slow(
        history={"step": 10, "current_regime_id": 0},
        regime_data={},
        model_state={},
    )
    assert result.regime_analysis.hmm_transition_detected is False


def test_think_slow_training_command_always_continue(agent):
    result = agent.think_slow(history={}, regime_data={}, model_state={})
    assert result.training_command == "CONTINUE"


def test_think_slow_predicted_regime_from_history(agent):
    result = agent.think_slow(
        history={"step": 1, "current_regime_id": 2},
        regime_data={},
        model_state={},
    )
    assert result.regime_analysis.predicted_regime == 2
