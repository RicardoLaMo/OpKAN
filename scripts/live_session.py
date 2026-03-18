import torch
import torch.nn as nn
import numpy as np
import time
import sys
from typing import Dict, Any

from src.data.parser import load_opra_data, clean_and_augment
from src.data.dataset import get_dataloader
from src.models.kan_layer import KANLayer, BSplineEdge
from src.models.heston_pde import heston_pde_loss, heston_boundary_loss
from src.models.hmm_regime import RegimeHMM
from src.engine.coordinator import EngineCoordinator
from src.agent.core import LiuClawAgent
from src.agent.fallback import RuleBasedFallbackAgent
from src.engine.telemetry import telemetry

# Ensure logs are visible in the launcher immediately
sys.stdout.reconfigure(line_buffering=True)

# ── Regime constants ────────────────────────────────────────────────────────
REGIME_LABELS   = {0: "STABLE", 1: "EXPANSION", 2: "JUMP"}
REGIME_WINDOW   = 40    # Reduced for faster demo feedback
REGIME_REFIT_INTERVAL = 20


class PIKANModel(nn.Module):
    def __init__(self, layers_config=None):
        super().__init__()
        if layers_config is None:
            layers_config = [3, 16, 1]
        self.layers = nn.ModuleList()
        for i in range(len(layers_config) - 1):
            self.layers.append(KANLayer(layers_config[i], layers_config[i + 1]))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# ── Model introspection helpers ─────────────────────────────────────────────

def extract_model_edge_stats(model: PIKANModel) -> Dict[str, Any]:
    """
    Return per-edge L1 norms for all BSplineEdge instances.
    Used by System 1 (think_fast) to decide which edges to prune.
    Format: {edge_id: {"l1_norm": float, "type": "bspline"|"pruned"}, ...}
    """
    stats: Dict[str, Any] = {}
    for l_idx, layer in enumerate(model.layers):
        for i in range(layer.in_features):
            for j in range(layer.out_features):
                edge = layer.edges[i][j]
                edge_id = f"L{l_idx}_N{i}_to_L{l_idx+1}_N{j}"
                if isinstance(edge, BSplineEdge):
                    l1 = edge.coefficients.detach().abs().sum().item()
                    stats[edge_id] = {"l1_norm": round(l1, 6), "type": "bspline"}
                else:
                    stats[edge_id] = {"l1_norm": 0.0, "type": "pruned"}
    return stats


def extract_model_state(model: PIKANModel) -> Dict[str, Any]:
    """
    Compact serializable summary of the full model topology.
    Used by System 2 (think_slow) for structural reasoning.
    Returns top-32 edges by L1 norm to stay within LLM context limits.
    """
    all_edges: Dict[str, Any] = {}
    for l_idx, layer in enumerate(model.layers):
        for i in range(layer.in_features):
            for j in range(layer.out_features):
                edge = layer.edges[i][j]
                edge_id = f"L{l_idx}_N{i}_to_L{l_idx+1}_N{j}"
                if isinstance(edge, BSplineEdge):
                    l1 = edge.coefficients.detach().abs().sum().item()
                    all_edges[edge_id] = {"type": "bspline", "l1_norm": round(l1, 6)}
                else:
                    all_edges[edge_id] = {"type": "pruned", "l1_norm": 0.0}

    # Truncate to top-32 by L1 norm (avoid bloating LLM prompt)
    sorted_edges = sorted(all_edges.items(), key=lambda kv: kv[1]["l1_norm"], reverse=True)
    return dict(sorted_edges[:32])


def _regime_stats_to_json(stats: dict) -> dict:
    """Convert numpy arrays in regime stats to plain Python lists for JSON serialization."""
    return {k: v.tolist() if hasattr(v, "tolist") else v for k, v in stats.items()}


# ── Agent selection ─────────────────────────────────────────────────────────

def _make_agent():
    """
    Try to connect to the vLLM server. Fall back to rule-based agent if unreachable.
    """
    try:
        agent = LiuClawAgent()
        # Probe using System 1 interface (minimal tokens)
        agent.think_fast(0, {}, 0.0)
        msg = "✅ vLLM server reachable — using real Qwen agents."
        print(msg)
        telemetry.log_event(msg)
        return agent
    except Exception as e:
        msg = f"⚠️  vLLM unreachable ({type(e).__name__}): using rule-based fallback agent."
        print(msg)
        telemetry.log_event(msg)
        return RuleBasedFallbackAgent()


# ── Main session ─────────────────────────────────────────────────────────────

def run_live_session(data_path: str, batch_size: int = 256, epochs: int = 1000):
    """
    Live end-to-end training session.
    - Real KAN model solving the Heston PDE via PINN.
    - Real Greeks via torch.autograd.
    - HMM regime detector updated every REGIME_REFIT_INTERVAL steps.
    - Dual-process coordinator backed by real Qwen agents (or rule-based fallback).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n--- ⚡ LIVE SESSION: OpKAN H200 Deployment --- Device: {device}")

    # Heston model parameters
    r, kappa, theta, sigma, rho, K, T = 0.05, 2.0, 0.04, 0.3, -0.7, 100.0, 1.0

    # Agent: real LLM or rule-based fallback
    agent = _make_agent()
    coordinator = EngineCoordinator(agent)
    coordinator.start_threads()

    # HMM regime detector
    hmm_regime = RegimeHMM(n_regimes=3)
    regime_feature_buffer = []       # rows of [pde_loss, abs(delta), abs(vega)]
    current_regime_id    = 0
    current_regime_label = "INITIALIZING"

    # Data loading
    print(f"📥 Loading data from {data_path}...")
    df = load_opra_data(data_path)
    if len(df) > 20000:
        print(f"💡 Subsampling to 20,000 rows for high-resolution TUI motion.")
        df = df.sample(20000).sort_values('timestamp')
    df = clean_and_augment(df)
    dataloader = get_dataloader(df, batch_size=batch_size, shuffle=False)

    # Model and optimizer
    model = PIKANModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(f"🔥 Starting Continuous Session: {len(df)} samples | {epochs} epochs | batch={batch_size}")
    total_steps = 0
    start_time  = time.time()
    loss_history = []   # rolling window of recent losses for System 2 context

    # Write initial state
    telemetry.write({
        "step": 0, "pde_loss": 1.0, "option_price": 10.0,
        "delta": 0.5, "gamma": 0.02, "vega": 0.1, "throughput": 0,
        "regime": current_regime_label, "logs": [],
        "s1_active": False, "s2_active": False, "dual_mode": True,
        "active": True,
    })

    try:
        for epoch in range(epochs):
            for i, (features, _) in enumerate(dataloader):
                total_steps += 1
                bs = features.shape[0]

                # PINN collocation points
                S_int = torch.rand(bs, 1, device=device, requires_grad=True) * 200.0
                v_int = torch.rand(bs, 1, device=device, requires_grad=True) * 0.5
                t_int = torch.rand(bs, 1, device=device, requires_grad=True) * T

                optimizer.zero_grad()

                V = model(torch.cat([S_int, v_int, t_int], dim=1))

                # PDE loss
                pde_loss = heston_pde_loss(
                    model, S_int, v_int, t_int, r, kappa, theta, sigma, rho
                )

                # Boundary conditions
                S_term = torch.rand(bs, 1, device=device) * 200.0
                v_term = torch.rand(bs, 1, device=device) * 0.5
                t_term = torch.ones(bs, 1, device=device) * T
                bnd_loss = heston_boundary_loss(
                    model, S_term, v_term, t_term, K,
                    torch.zeros(bs, 1, device=device), v_term,
                    torch.rand(bs, 1, device=device) * T,
                    torch.ones(bs, 1, device=device) * 1000.0, v_term,
                    torch.rand(bs, 1, device=device) * T,
                    r, T,
                )

                # ── Sparsity Penalty (L1) ──
                # Penalize B-spline coefficients to drive stagnant edges toward zero
                l1_penalty = 0.0
                for layer in model.layers:
                    for i in range(layer.in_features):
                        for j in range(layer.out_features):
                            edge = layer.edges[i][j]
                            if isinstance(edge, BSplineEdge):
                                l1_penalty += edge.coefficients.abs().sum()
                
                # Combine losses with a moderate penalty for L1 in demo
                loss = pde_loss + bnd_loss + (0.005 * l1_penalty)
                loss.backward()

                # 🚀 Real-time Greeks via autograd (Robust)
                if V.requires_grad:
                    dV_dS      = torch.autograd.grad(V.sum(), S_int, create_graph=True, retain_graph=True)[0]
                    delta_val  = dV_dS.mean().item()
                    gamma_val  = torch.autograd.grad(dV_dS.sum(), S_int, retain_graph=True)[0].mean().item()
                    vega_val   = torch.autograd.grad(V.sum(), v_int, retain_graph=True)[0].mean().item()
                else:
                    delta_val, gamma_val, vega_val = 0.0, 0.0, 0.0

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                loss_val = loss.item()
                loss_history.append(loss_val)
                if len(loss_history) > 50:
                    loss_history.pop(0)

                # ── HMM regime update (Trigger-based) ──────────────────────
                regime_feature_buffer.append([loss_val, abs(delta_val), abs(vega_val)])
                
                # Dynamic Trigger: Force refit if loss spikes 2x above moving average
                loss_ma = np.mean(loss_history) if loss_history else loss_val
                loss_spike = loss_val > (loss_ma * 2.0) and total_steps > REGIME_WINDOW

                if (
                    (total_steps % REGIME_REFIT_INTERVAL == 0 or loss_spike)
                    and len(regime_feature_buffer) >= REGIME_WINDOW
                ):
                    try:
                        window = np.array(regime_feature_buffer[-REGIME_WINDOW:])
                        hmm_regime.fit(window)
                        labels = hmm_regime.predict_regimes(window)

                        # Sort states by mean loss (col 0): ascending → 0=STABLE, 2=JUMP
                        means = hmm_regime.model.means_[:, 0]
                        sorted_states = np.argsort(means)
                        remap = {int(old): int(new) for new, old in enumerate(sorted_states)}
                        
                        new_regime_id = remap[int(labels[-1])]
                        if new_regime_id != current_regime_id:
                            msg = f"🚨 TRIGGER: Regime shift detected {REGIME_LABELS.get(current_regime_id)} -> {REGIME_LABELS.get(new_regime_id)}"
                            print(msg)
                            telemetry.log_event(msg)
                            
                        current_regime_id    = new_regime_id
                        current_regime_label = REGIME_LABELS[current_regime_id]
                    except Exception as hmm_err:
                        telemetry.log_event(f"HMM refit failed at step {total_steps}: {hmm_err}")

                # ── LLM / fallback agent interaction ──────────────────────
                coordinator.apply_pending_mutations(model, optimizer)

                if total_steps % 10 == 0:
                    coordinator.request_reflex(
                        total_steps,
                        extract_model_edge_stats(model),
                        loss_val,
                    )

                if total_steps % 50 == 0:
                    regime_data = (
                        _regime_stats_to_json(hmm_regime.get_regime_stats())
                        if hmm_regime.is_fitted
                        else {}
                    )
                    coordinator.request_strategic(
                        {
                            "step": total_steps,
                            "loss": loss_val,
                            "loss_history": loss_history[-10:],
                            "current_regime_id": current_regime_id,
                        },
                        regime_data,
                        extract_model_state(model),
                    )

                # ── Telemetry publish ──────────────────────────────────────
                elapsed = time.time() - start_time
                tput = int((total_steps * batch_size) / elapsed)
                
                # Fetch only new logs from the global store to avoid overwriting
                current_tel = telemetry.read()
                
                telemetry.write({
                    "step":         total_steps,
                    "pde_loss":     loss_val,
                    "option_price": V.mean().item(),
                    "delta":        delta_val,
                    "gamma":        gamma_val,
                    "vega":         vega_val,
                    "throughput":   tput,
                    "regime":       current_regime_label,
                    "logs":         current_tel.get("logs", []),
                    "s1_active":    coordinator._reflex_thread.is_alive() if coordinator._reflex_thread else False,
                    "s2_active":    coordinator._strategic_thread.is_alive() if coordinator._strategic_thread else False,
                    "dual_mode":    True,
                    "active":       True,
                })

                if total_steps % 50 == 0:
                    print(f"[Step {total_steps}] Loss: {loss_val:.6f} | Regime: {current_regime_label}")

    finally:
        final_data = telemetry.read()
        final_data["active"] = False
        telemetry.write(final_data)
        coordinator.stop_threads()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="OpKAN Live Session")
    parser.add_argument("--data_path", type=str, default="data/real_market_sim.csv")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=1000)
    args = parser.parse_args()
    
    run_live_session(args.data_path, args.batch_size, args.epochs)
