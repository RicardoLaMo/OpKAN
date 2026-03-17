import torch
import torch.nn as nn
import time
from src.agent.core import LiuClawAgent
from src.agent.dsl import StrategicDecision, ReflexDecision, RegimeThesis, EdgeMutation
from src.engine.coordinator import EngineCoordinator
from src.models.kan_layer import KANLayer
from src.models.heston_pde import heston_pde_loss
from src.data.parser import load_opra_data, clean_and_augment
from src.data.dataset import get_dataloader

class PIKANModel(nn.Module):
    def __init__(self, layers_config=[3, 16, 1]):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers_config)-1):
            self.layers.append(KANLayer(layers_config[i], layers_config[i+1]))
    def forward(self, x):
        for layer in self.layers: x = layer(x)
        return x

def run_experiment(mode: str, data_path: str, epochs: int = 2):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n--- Experiment Mode: {mode} ---")
    
    # Setup
    df = load_opra_data(data_path)
    df = clean_and_augment(df)
    dataloader = get_dataloader(df, batch_size=4096, shuffle=False)
    model = PIKANModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    agent = LiuClawAgent()
    coordinator = EngineCoordinator(agent)
    
    # Mocking agent responses for deterministic benchmark
    if mode != "baseline":
        def mock_fast(step, stats, delta):
            return ReflexDecision(reasoning="Reflexive prune", prunes=[], lr_adjustment=1.0)
        def mock_slow(history, regime, state):
            return StrategicDecision(
                reasoning="Strategic replace",
                mutations=[EdgeMutation(edge_id="L0_N0_to_L1_N0", action="REPLACE", formula="torch.pow(x, 2)", reasoning="Q")],
                regime_analysis=RegimeThesis(hmm_transition_detected=False)
            )
        agent.think_fast = mock_fast
        agent.think_slow = mock_slow
        coordinator.start_threads()

    start_time = time.time()
    total_steps = 0
    
    for epoch in range(epochs):
        for i, (features, labels) in enumerate(dataloader):
            total_steps += 1
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            S, v, t = features[:, 0:1], features[:, 1:2], features[:, 2:3]
            S.requires_grad_(True); v.requires_grad_(True); t.requires_grad_(True)
            loss = heston_pde_loss(model, S, v, t, 0.05, 2.0, 0.04, 0.3, -0.7)
            loss.backward()
            optimizer.step()
            
            if mode == "dual":
                coordinator.apply_pending_mutations(model, optimizer)
                if total_steps % 50 == 0:
                    coordinator.request_reflex(total_steps, {}, 0.0)
                if total_steps % 200 == 0:
                    coordinator.request_strategic({}, {}, {})
            
    if mode != "baseline": coordinator.stop_threads()
    
    duration = time.time() - start_time
    print(f"Result: {mode} finished in {duration:.2f}s | Final Loss: {loss.item():.6f}")
    return duration, loss.item()

if __name__ == "__main__":
    data_path = "data/real_market_sim.csv"
    run_experiment("baseline", data_path)
    run_experiment("dual", data_path)
