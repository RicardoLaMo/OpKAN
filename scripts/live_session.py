import torch
import torch.nn as nn
import time
from src.data.parser import load_opra_data, clean_and_augment
from src.data.dataset import get_dataloader
from src.models.kan_layer import KANLayer
from src.models.heston_pde import heston_pde_loss, heston_boundary_loss
from src.engine.coordinator import EngineCoordinator
from src.agent.core import LiuClawAgent
from src.agent.dsl import StrategicDecision, ReflexDecision, EdgeMutation, RegimeThesis


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
    Uses random PDE collocation points (PINN approach) + real market data for step pacing.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n--- LIVE SESSION: OpKAN Deployment --- Device: {device}")

    # Heston model parameters
    r, kappa, theta, sigma, rho, K, T = 0.05, 2.0, 0.04, 0.3, -0.7, 100.0, 1.0

    # 1. Mock agent using current dual-process API
    agent = LiuClawAgent()

    def mock_think_fast(step, edge_stats, loss_delta):
        time.sleep(0.05)
        return ReflexDecision(
            reasoning=f"Step {step}: loss_delta={loss_delta:.4f}, no pruning needed.",
            prunes=[],
            lr_adjustment=1.0,
        )

    def mock_think_slow(history, regime_data, model_state):
        time.sleep(0.2)
        step = history.get("step", 0)
        regime = 1 if step > 500 else 0
        return StrategicDecision(
            reasoning=f"Step {step}: analyzing volatility topology.",
            mutations=[EdgeMutation(
                edge_id="L0_N0_to_L1_N0",
                action="REPLACE",
                formula="torch.pow(x, 2)",
                reasoning="Quadratic fit for spot price curvature.",
            )],
            regime_analysis=RegimeThesis(
                hmm_transition_detected=(regime == 1),
                predicted_regime=regime,
                thesis_statement="Volatility expansion detected in synthetic regime.",
            ),
            training_command="CONTINUE",
        )

    agent.think_fast = mock_think_fast
    agent.think_slow = mock_think_slow

    coordinator = EngineCoordinator(agent)
    coordinator.start_threads()

    # 2. Data loading (drives step count and real-data regime analysis)
    print(f"Loading data from {data_path}...")
    df = load_opra_data(data_path)
    df = clean_and_augment(df)
    dataloader = get_dataloader(df, batch_size=batch_size, shuffle=False)

    # 3. Model and optimizer
    model = PIKANModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(f"Starting: {len(df)} samples | {epochs} epochs | batch={batch_size}")
    total_steps = 0
    start_time = time.time()

    for epoch in range(epochs):
        for i, (features, _) in enumerate(dataloader):
            total_steps += 1
            bs = features.shape[0]

            # --- PINN: random collocation points over PDE domain ---
            S_int = torch.rand(bs, 1, device=device, requires_grad=True) * 200.0
            v_int = torch.rand(bs, 1, device=device, requires_grad=True) * 0.5
            t_int = torch.rand(bs, 1, device=device, requires_grad=True) * T

            # Boundary collocation points
            S_term = torch.rand(bs, 1, device=device) * 200.0
            v_term = torch.rand(bs, 1, device=device) * 0.5
            t_term = torch.ones(bs, 1, device=device) * T
            S_zero = torch.zeros(bs, 1, device=device)
            v_zero = torch.rand(bs, 1, device=device) * 0.5
            t_zero = torch.rand(bs, 1, device=device) * T
            S_inf = torch.ones(bs, 1, device=device) * 1000.0
            v_inf = torch.rand(bs, 1, device=device) * 0.5
            t_inf = torch.rand(bs, 1, device=device) * T

            optimizer.zero_grad()

            pde_loss = heston_pde_loss(model, S_int, v_int, t_int, r, kappa, theta, sigma, rho)
            bnd_loss = heston_boundary_loss(
                model, S_term, v_term, t_term, K,
                S_zero, v_zero, t_zero,
                S_inf, v_inf, t_inf, r, T
            )
            loss = pde_loss + bnd_loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # LLM agent interaction
            coordinator.apply_pending_mutations(model, optimizer)

            if total_steps % 10 == 0:
                coordinator.request_reflex(
                    step=total_steps,
                    edge_stats={"L0_N0_to_L1_N0": {"l1": 0.5}},
                    loss_delta=loss.item(),
                )
            if total_steps % 50 == 0:
                coordinator.request_strategic(
                    history={"step": total_steps, "loss": loss.item()},
                    regime_data={"regime": 0},
                    model_state={"layers": len(model.layers)},
                )

            if i % 50 == 0:
                elapsed = time.time() - start_time
                tput = (total_steps * batch_size) / elapsed
                print(f"[Epoch {epoch+1}] Step {i}/{len(dataloader)} | "
                      f"PDE: {pde_loss.item():.4f} | Bnd: {bnd_loss.item():.4f} | "
                      f"Throughput: {tput:.0f} samp/s")

    total_duration = time.time() - start_time
    throughput = (len(df) * epochs) / total_duration
    print(f"\n--- Stress Test Complete ---")
    print(f"Total Time: {total_duration:.2f}s | Throughput: {throughput:.2f} samp/s")
    print(f"Final Loss: {loss.item():.6f}")

    coordinator.stop_threads()


if __name__ == "__main__":
    run_live_session("data/real_market_sim.csv")
