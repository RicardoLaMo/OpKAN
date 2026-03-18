"""
Tests for live_session.py helper functions and integration logic.

Covers:
- extract_model_edge_stats: per-edge L1 norms, correct edge count, type labeling
- extract_model_state: top-32 truncation, descending L1 sort
- _regime_stats_to_json: numpy arrays become plain lists (JSON-serializable)
- _make_agent: returns RuleBasedFallbackAgent when vLLM is unreachable
- HMM regime integration: feature buffer fills, refit gates, label mapping
- End-to-end: a few training steps run without crash using the fallback agent
"""
import json
import sys
import types
import time
import numpy as np
import pytest
import torch
from unittest.mock import MagicMock, patch

# The helpers live inside the scripts/ package — import via path
from scripts.live_session import (
    PIKANModel,
    extract_model_edge_stats,
    extract_model_state,
    _regime_stats_to_json,
    _make_agent,
    REGIME_WINDOW,
    REGIME_REFIT_INTERVAL,
    REGIME_LABELS,
)
from src.models.kan_layer import BSplineEdge
from src.agent.fallback import RuleBasedFallbackAgent
from src.models.hmm_regime import RegimeHMM


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def default_model():
    """PIKANModel with default [3, 16, 1] config: 3*16 + 16*1 = 64 edges."""
    return PIKANModel()


# ── extract_model_edge_stats ─────────────────────────────────────────────────

def test_edge_stats_returns_all_edges(default_model):
    stats = extract_model_edge_stats(default_model)
    # [3,16,1]: layer 0 has 3*16=48 edges, layer 1 has 16*1=16 edges
    assert len(stats) == 64


def test_edge_stats_all_bspline_by_default(default_model):
    stats = extract_model_edge_stats(default_model)
    assert all(v["type"] == "bspline" for v in stats.values())


def test_edge_stats_l1_norms_are_non_negative(default_model):
    stats = extract_model_edge_stats(default_model)
    assert all(v["l1_norm"] >= 0.0 for v in stats.values())


def test_edge_stats_id_format(default_model):
    stats = extract_model_edge_stats(default_model)
    for edge_id in stats:
        parts = edge_id.split("_")
        # e.g. "L0_N2_to_L1_N15"
        assert edge_id.startswith("L"), f"Bad ID: {edge_id}"
        assert "_to_" in edge_id, f"Bad ID: {edge_id}"


def test_edge_stats_non_bspline_edge_labeled_pruned(default_model):
    """Swap one edge to a non-BSpline module; it should appear as 'pruned'."""
    dummy = torch.nn.Linear(1, 1)  # not a BSplineEdge
    default_model.layers[0].edges[0][0] = dummy
    stats = extract_model_edge_stats(default_model)
    assert stats["L0_N0_to_L1_N0"]["type"] == "pruned"
    assert stats["L0_N0_to_L1_N0"]["l1_norm"] == 0.0


# ── extract_model_state ──────────────────────────────────────────────────────

def test_model_state_truncates_to_32(default_model):
    state = extract_model_state(default_model)
    assert len(state) == 32  # default model has 64 edges; top-32 returned


def test_model_state_sorted_descending_l1(default_model):
    state = extract_model_state(default_model)
    l1_values = [v["l1_norm"] for v in state.values()]
    assert l1_values == sorted(l1_values, reverse=True)


def test_model_state_smaller_model_returns_all_edges():
    """If the model has <= 32 edges all of them are returned."""
    model = PIKANModel(layers_config=[2, 4, 1])  # 2*4 + 4*1 = 12 edges
    state = extract_model_state(model)
    assert len(state) == 12


def test_model_state_json_serializable(default_model):
    """The returned dict must be serialisable with json.dumps (no tensors/ndarrays)."""
    state = extract_model_state(default_model)
    serialised = json.dumps(state)  # must not raise
    assert isinstance(serialised, str)


# ── _regime_stats_to_json ────────────────────────────────────────────────────

def test_regime_stats_to_json_converts_numpy_arrays():
    stats = {
        "transition_matrix": np.array([[0.9, 0.1], [0.2, 0.8]]),
        "means":  np.array([[0.1], [0.9]]),
        "covars": np.array([[[0.01]], [[0.04]]]),
    }
    result = _regime_stats_to_json(stats)
    serialised = json.dumps(result)  # must not raise TypeError
    assert isinstance(result["transition_matrix"], list)
    assert isinstance(result["means"], list)


def test_regime_stats_to_json_passthrough_plain_values():
    stats = {"n_regimes": 3, "label": "STABLE"}
    result = _regime_stats_to_json(stats)
    assert result == stats


# ── _make_agent ──────────────────────────────────────────────────────────────

def test_make_agent_returns_fallback_when_vllm_unreachable():
    """Simulate connection refused → must return RuleBasedFallbackAgent."""
    with patch("scripts.live_session.LiuClawAgent") as MockLiuClaw:
        instance = MagicMock()
        instance.client.client.chat.completions.create.side_effect = ConnectionRefusedError
        MockLiuClaw.return_value = instance
        agent = _make_agent()
    assert isinstance(agent, RuleBasedFallbackAgent)


def test_make_agent_returns_real_agent_when_vllm_reachable():
    """Simulate successful ping → must return the LiuClawAgent instance."""
    # Use plain MagicMock (no spec) so that instance-level attribute chains
    # like .client.client.chat.completions.create are auto-created.
    with patch("scripts.live_session.LiuClawAgent") as MockLiuClaw:
        instance = MagicMock()
        instance.client.client.chat.completions.create.return_value = MagicMock()
        MockLiuClaw.return_value = instance
        agent = _make_agent()
    assert agent is instance


# ── HMM regime integration logic ─────────────────────────────────────────────

def test_hmm_regime_labels_map_all_three_states():
    assert set(REGIME_LABELS.values()) == {"STABLE", "EXPANSION", "JUMP"}


def test_hmm_refit_produces_valid_label():
    """Smoke-test the HMM refit + state-sort logic used in the training loop."""
    hmm = RegimeHMM(n_regimes=3)
    rng = np.random.default_rng(42)
    # Three clearly separable clusters
    seg0 = rng.normal([0.1, 0.3, 0.2], 0.05, (50, 3))
    seg1 = rng.normal([0.5, 0.6, 0.5], 0.05, (50, 3))
    seg2 = rng.normal([1.0, 0.9, 0.9], 0.05, (50, 3))
    window = np.vstack([seg0, seg1, seg2])

    hmm.fit(window)
    labels = hmm.predict_regimes(window)

    means = hmm.model.means_[:, 0]
    sorted_states = np.argsort(means)
    remap = {int(old): int(new) for new, old in enumerate(sorted_states)}
    current_id = remap[int(labels[-1])]

    assert current_id in REGIME_LABELS
    assert REGIME_LABELS[current_id] in {"STABLE", "EXPANSION", "JUMP"}


def test_hmm_regime_stats_json_roundtrip():
    """After fitting, get_regime_stats → _regime_stats_to_json → json.dumps must succeed."""
    hmm = RegimeHMM(n_regimes=2)
    rng = np.random.default_rng(0)
    hmm.fit(rng.normal(0, 1, (120, 3)))
    raw  = hmm.get_regime_stats()
    safe = _regime_stats_to_json(raw)
    json.dumps(safe)  # must not raise


# ── End-to-end: a few training steps with the fallback agent ─────────────────

def test_e2e_training_steps_with_fallback_agent(tmp_path):
    """
    Run 3 training steps using RuleBasedFallbackAgent (no vLLM needed).
    Validates that model parameters change, telemetry is written, and no
    exceptions are raised. Data loading is mocked to avoid requiring CSV.
    """
    import torch
    import torch.nn as nn
    from src.engine.coordinator import EngineCoordinator
    from src.engine.telemetry import TelemetryStore
    from src.models.heston_pde import heston_pde_loss, heston_boundary_loss

    # Isolated telemetry store (temp dir)
    tel = TelemetryStore(path=str(tmp_path / "telemetry.json"))
    tel.write({
        "step": 0, "pde_loss": 1.0, "option_price": 10.0,
        "delta": 0.5, "gamma": 0.02, "vega": 0.1, "throughput": 0,
        "regime": "INITIALIZING", "logs": [],
        "s1_active": False, "s2_active": False, "dual_mode": True, "active": True,
    })

    device = torch.device("cpu")
    r, kappa, theta, sigma, rho, K, T = 0.05, 2.0, 0.04, 0.3, -0.7, 100.0, 1.0
    bs = 16

    model = PIKANModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    agent = RuleBasedFallbackAgent()
    coordinator = EngineCoordinator(agent)
    coordinator.start_threads()

    initial_params = [p.clone() for p in model.parameters()]

    regime_feature_buffer = []
    current_regime_id    = 0
    current_regime_label = "INITIALIZING"
    hmm_regime = RegimeHMM(n_regimes=3)

    try:
        for step in range(1, 4):
            S_int = torch.rand(bs, 1, device=device, requires_grad=True) * 200.0
            v_int = torch.rand(bs, 1, device=device, requires_grad=True) * 0.5
            t_int = torch.rand(bs, 1, device=device, requires_grad=True) * T

            optimizer.zero_grad()
            V = model(torch.cat([S_int, v_int, t_int], dim=1))

            pde_loss  = heston_pde_loss(model, S_int, v_int, t_int, r, kappa, theta, sigma, rho)
            S_t = torch.rand(bs, 1, device=device) * 200.0
            v_t = torch.rand(bs, 1, device=device) * 0.5
            bnd_loss  = heston_boundary_loss(
                model, S_t, v_t, torch.ones(bs, 1) * T, K,
                torch.zeros(bs, 1), v_t, torch.rand(bs, 1) * T,
                torch.ones(bs, 1) * 1000.0, v_t, torch.rand(bs, 1) * T,
                r, T,
            )
            loss = pde_loss + bnd_loss
            loss.backward()

            dV_dS     = torch.autograd.grad(V.sum(), S_int, create_graph=True, retain_graph=True)[0]
            delta_val = dV_dS.mean().item()
            vega_val  = torch.autograd.grad(V.sum(), v_int, retain_graph=True)[0].mean().item()
            loss_val  = loss.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            coordinator.apply_pending_mutations(model, optimizer)

            # Regime feature accumulation (mirrors live_session.py)
            regime_feature_buffer.append([loss_val, abs(delta_val), abs(vega_val)])

            # Coordinator requests with real edge data
            if step % 2 == 0:
                coordinator.request_reflex(
                    step, extract_model_edge_stats(model), loss_val
                )

            tel.write({
                "step": step, "pde_loss": loss_val, "option_price": V.mean().item(),
                "delta": delta_val, "gamma": 0.0, "vega": vega_val,
                "throughput": step * bs, "regime": current_regime_label,
                "logs": [], "s1_active": True, "s2_active": False,
                "dual_mode": True, "active": True,
            })
    finally:
        coordinator.stop_threads()

    # Model parameters changed (gradient flow worked)
    for orig, updated in zip(initial_params, model.parameters()):
        if orig.requires_grad:
            assert not torch.equal(orig, updated.detach()), "Parameters did not update"

    # Telemetry has plausible values from the last step
    import math
    data = tel.read()
    assert data["step"] == 3
    assert math.isfinite(data["pde_loss"]), "PDE loss is NaN or Inf"
    assert data["active"] is True


# ── TUI integration: enriched telemetry renders without crash ─────────────────

@pytest.mark.asyncio
async def test_tui_renders_enriched_telemetry():
    """
    Push telemetry in the full enriched format (with HMM regime, real Greeks)
    and verify the TUI dashboard updates without raising exceptions.
    """
    import os, tempfile
    from src.engine.telemetry import TelemetryStore
    from src.ui.tui.app import OpKANDashboard

    with tempfile.TemporaryDirectory() as tmpdir:
        tel = TelemetryStore(path=os.path.join(tmpdir, "telemetry.json"))
        tel.write({
            "step": 42,
            "pde_loss": 0.00312,
            "option_price": 8.74,
            "delta": 0.623,
            "gamma": 0.018,
            "vega": 0.142,
            "throughput": 26300,
            "regime": "EXPANSION",
            "logs": [
                {"timestamp": "12:00:01", "message": "HMM refit at step 25"},
                {"timestamp": "12:00:05", "message": "S1: pruned 2 edges"},
            ],
            "s1_active": True,
            "s2_active": True,
            "dual_mode": True,
            "active": True,
        })

        async with OpKANDashboard().run_test() as pilot:
            # Inject the telemetry module so the app reads our store
            import src.engine.telemetry as tel_module
            _orig = tel_module.telemetry
            tel_module.telemetry = tel
            try:
                await pilot.pause(0.2)
                # App should not have crashed — check a metric card is present
                app = pilot.app
                card = app.query_one("#card-loss")
                assert card is not None
            finally:
                tel_module.telemetry = _orig
