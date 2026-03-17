import torch
import torch.nn as nn
import time
import os
import json
from src.data.parser import load_opra_data, clean_and_augment
from src.data.dataset import get_dataloader
from src.models.kan_layer import KANLayer
from src.models.heston_pde import heston_pde_loss
from src.engine.coordinator import EngineCoordinator
from src.agent.core import LiuClawAgent
from src.agent.dsl import LiuClawDecision, EdgeMutation, RegimeThesis

class PIKANModel(nn.Module):
    def __init__(self, layers_config=[3, 16, 1]):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers_config)-1):
            self.layers.append(KANLayer(layers_config[i], layers_config[i+1]))
            
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def run_live_session(data_path: str, batch_size: int = 4096, epochs: int = 5):
    """
    Simulates a live end-to-end training session on H200.
    Stress tests throughput and async LLM interaction.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n--- ⚡ LIVE SESSION: OpKAN H200 Deployment --- Device: {device}")
    
    # 1. Start vLLM Mock (Real vLLM requires background process setup)
    # For this session, we'll use a high-frequency mock that simulates vLLM latency
    agent = LiuClawAgent()
    # Simulated vLLM fast response (0.2s)
    def mock_vllm_response(kan_state, health):
        time.sleep(0.2) 
        # Deterministic simulation of a regime shift detection
        step = kan_state.get("step", 0)
        regime = 1 if step > 500 else 0
        return LiuClawDecision(
            training_command="CONTINUE",
            reasoning=f"Analyzed state at step {step}. Detecting volatility expansion.",
            mutations=[EdgeMutation(
                edge_id="L0_N0_to_L1_N0",
                action="REPLACE",
                formula="torch.pow(x, 2)",
                reasoning="Quadratic fit for spot price."
            )],
            regime_analysis=RegimeThesis(
                hmm_transition_detected=(regime == 1),
                predicted_regime=regime,
                thesis_statement="Vol expansion triggered by synthetic high-fidelity returns."
            ),
            confidence=0.98
        )
    agent.decide_mutations = mock_vllm_response
    
    coordinator = EngineCoordinator(agent)
    coordinator.start_agent_thread()
    
    # 2. Data Loading
    print("📥 Loading real_market_sim.csv (1M rows)...")
    df = load_opra_data(data_path)
    df = clean_and_augment(df)
    dataloader = get_dataloader(df, batch_size=batch_size, shuffle=False) # Sequential for regime demo
    
    # 3. Model & PDE Config
    model = PIKANModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    r, kappa, theta, sigma, rho = 0.05, 2.0, 0.04, 0.3, -0.7
    
    # 4. Stress Test Training Loop
    print(f"🔥 Starting Stress Test: {len(df)} samples | {epochs} epochs | Batch Size: {batch_size}")
    total_steps = 0
    start_time = time.time()
    
    for epoch in range(epochs):
        for i, (features, labels) in enumerate(dataloader):
            total_steps += 1
            features, labels = features.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            S, v, t = features[:, 0:1], features[:, 1:2], features[:, 2:3]
            S.requires_grad_(True); v.requires_grad_(True); t.requires_grad_(True)
            
            # PDE Math
            loss = heston_pde_loss(model, S, v, t, r, kappa, theta, sigma, rho)
            loss.backward()
            optimizer.step()
            
            # 🚀 LLM/KAN Interaction Check
            pre_loss = loss.item()
            status = coordinator.apply_pending_mutations(model)
            if status == "HALT": break
            
            # Post-mutation loss (optional check for reward)
            # In a real loop, we might wait a few steps to see improvement
            
            # Stress the agent with requests every 10 steps
            if total_steps % 10 == 0:
                context = {"step": total_steps, "kan_state": "extracted_at_runtime"}
                health = {"throughput": "high"}
                coordinator.request_mutation(context, health)
                
                # Log for LoRA (using mock decision for this trace)
                from src.agent.data_collector import collector
                # collector.log_trajectory(context, {"mock": "decision"}, pre_loss, pre_loss * 0.9)
            
            if i % 50 == 0:
                print(f"[Epoch {epoch+1}] Step {i}/{len(dataloader)} | Loss: {loss.item():.6f} | Mut Queue: {coordinator.agent_thread.is_alive()}")

    total_duration = time.time() - start_time
    throughput = (len(df) * epochs) / total_duration
    print(f"\n--- ✅ Stress Test Complete ---")
    print(f"Total Time: {total_duration:.2f}s")
    print(f"Throughput: {throughput:.2f} samples/sec")
    print(f"Final Loss: {loss.item():.6f}")
    
    coordinator.stop_agent_thread()

if __name__ == "__main__":
    run_live_session("data/real_market_sim.csv")
