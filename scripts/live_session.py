import torch
import torch.nn as nn
import time
import random
from src.data.parser import load_opra_data, clean_and_augment
from src.data.dataset import get_dataloader
from src.models.kan_layer import KANLayer
from src.models.heston_pde import heston_pde_loss, heston_boundary_loss
from src.engine.coordinator import EngineCoordinator
from src.agent.core import LiuClawAgent
from src.agent.dsl import StrategicDecision, ReflexDecision, EdgeMutation, RegimeThesis
from src.engine.telemetry import telemetry


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


def run_live_session(data_path: str, batch_size: int = 4096, epochs: int = 5):
    """
    Live end-to-end training session stress-testing throughput and LLM interaction.
    Integrates real-time Greeks calculation and Telemetry publishing.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n--- ⚡ LIVE SESSION: OpKAN H200 Deployment --- Device: {device}")

    # Heston model parameters
    r, kappa, theta, sigma, rho, K, T = 0.05, 2.0, 0.04, 0.3, -0.7, 100.0, 1.0

    # 1. Start vLLM Mock
    agent = LiuClawAgent()

    def mock_think_fast(step, edge_stats, loss_delta):
        time.sleep(0.02)
        return ReflexDecision(
            reasoning=f"Reflexive prune at step {step}.",
            prunes=[f"L0_N{random.randint(0,15)}_to_L1_N{random.randint(0,15)}"],
            lr_adjustment=1.0,
        )

    def mock_think_slow(history, regime_data, model_state):
        time.sleep(0.2)
        step = history.get("step", 0)
        regime = 1 if step > 500 else 0
        return StrategicDecision(
            reasoning=f"Strategic review at step {step}. Market state: {regime}.",
            mutations=[EdgeMutation(
                edge_id="L0_N0_to_L1_N0",
                action="REPLACE",
                formula="torch.pow(x, 2)",
                reasoning="Capturing volatility smile curvature.",
            )],
            regime_analysis=RegimeThesis(
                hmm_transition_detected=(regime == 1),
                predicted_regime=regime,
                thesis_statement="Structural shift detected via HMM transition probabilities."
            ),
            training_command="CONTINUE",
        )

    agent.think_fast = mock_think_fast
    agent.think_slow = mock_think_slow

    coordinator = EngineCoordinator(agent)
    coordinator.start_threads()

    # 2. Data loading
    print(f"📥 Loading data from {data_path}...")
    df = load_opra_data(data_path)
    df = clean_and_augment(df)
    dataloader = get_dataloader(df, batch_size=batch_size, shuffle=False)

    # 3. Model and optimizer
    model = PIKANModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(f"🔥 Starting Stress Test: {len(df)} samples | {epochs} epochs | batch={batch_size}")
    total_steps = 0
    start_time = time.time()

    # Write initial state
    telemetry.write({
        "step": 0, "pde_loss": 1.0, "option_price": 10.0,
        "delta": 0.5, "gamma": 0.02, "vega": 0.1, "throughput": 0,
        "regime": "INITIALIZING", "logs": [], "s1_active": False, "s2_active": False, "dual_mode": True
    })

    for epoch in range(epochs):
        for i, (features, _) in enumerate(dataloader):
            total_steps += 1
            bs = features.shape[0]

            # PINN collocation points
            S_int = torch.rand(bs, 1, device=device, requires_grad=True) * 200.0
            v_int = torch.rand(bs, 1, device=device, requires_grad=True) * 0.5
            t_int = torch.rand(bs, 1, device=device, requires_grad=True) * T

            optimizer.zero_grad()

            # Forward for Greeks
            # Input: (S, v, t)
            inputs = torch.cat([S_int, v_int, t_int], dim=1)
            V = model(inputs)
            
            # PDE Loss
            pde_loss = heston_pde_loss(model, S_int, v_int, t_int, r, kappa, theta, sigma, rho)
            
            # Boundary conditions
            S_term = torch.rand(bs, 1, device=device) * 200.0
            v_term = torch.rand(bs, 1, device=device) * 0.5
            t_term = torch.ones(bs, 1, device=device) * T
            bnd_loss = heston_boundary_loss(model, S_term, v_term, t_term, K,
                                            torch.zeros(bs, 1, device=device), v_term, torch.rand(bs, 1, device=device)*T,
                                            torch.ones(bs, 1, device=device)*1000.0, v_term, torch.rand(bs, 1, device=device)*T,
                                            r, T)
            
            loss = pde_loss + bnd_loss
            loss.backward()
            optimizer.step()

            # 🚀 Real-time Greeks Extraction
            dV_dS = torch.autograd.grad(V.sum(), S_int, retain_graph=True)[0]
            delta_val = dV_dS.mean().item()
            gamma_val = torch.autograd.grad(dV_dS.sum(), S_int, retain_graph=True)[0].mean().item()
            vega_val = torch.autograd.grad(V.sum(), v_int, retain_graph=True)[0].mean().item()

            # LLM interaction
            coordinator.apply_pending_mutations(model, optimizer)

            if total_steps % 10 == 0:
                coordinator.request_reflex(total_steps, {"l1_norm": 0.001}, loss.item())
            if total_steps % 100 == 0:
                coordinator.request_strategic({"step": total_steps}, {"regime": 0}, {"model": "pi-kan"})

            # 📈 Publish Telemetry
            if total_steps % 2 == 0:
                elapsed = time.time() - start_time
                tput = (total_steps * batch_size) / elapsed
                telemetry.write({
                    "step": total_steps,
                    "pde_loss": loss.item(),
                    "option_price": V.mean().item(),
                    "delta": delta_val,
                    "gamma": gamma_val,
                    "vega": vega_val,
                    "throughput": int(tput),
                    "regime": "EXPANSION" if total_steps > 500 else "STABLE",
                    "logs": telemetry.read().get("logs", []),
                    "s1_active": coordinator._reflex_thread.is_alive() if coordinator._reflex_thread else False,
                    "s2_active": coordinator._strategic_thread.is_alive() if coordinator._strategic_thread else False,
                    "dual_mode": True
                })

            if i % 50 == 0:
                print(f"[Epoch {epoch+1}] Step {i}/{len(dataloader)} | Loss: {loss.item():.6f}")

    coordinator.stop_threads()


if __name__ == "__main__":
    run_live_session("data/real_market_sim.csv")
