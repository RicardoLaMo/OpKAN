import torch
import torch.nn as nn
import time
import pandas as pd
import numpy as np
from src.data.parser import load_opra_data, clean_and_augment
from src.data.dataset import get_dataloader
from src.models.kan_layer import KANLayer
from src.models.heston_pde import heston_pde_loss, heston_boundary_loss
from src.engine.coordinator import EngineCoordinator
from src.agent.core import LiuClawAgent
from src.agent.dsl import ReflexDecision, StrategicDecision, RegimeThesis
from src.config import load_config, get_heston_params, get_collocation_params, get_training_params

class PIKANModel(nn.Module):
    """
    Complete PI-KAN model for Heston PDE.
    (S, v, t) -> V
    """
    def __init__(self, layers_config=[3, 16, 1]):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers_config)-1):
            self.layers.append(KANLayer(layers_config[i], layers_config[i+1]))
            
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def benchmark_h200(data_path: str, batch_size: int = None, epochs: int = None):
    cfg = load_config()
    h = get_heston_params(cfg)
    c = get_collocation_params(cfg)
    t = get_training_params(cfg)

    if batch_size is None:
        batch_size = t['batch_size']
    if epochs is None:
        epochs = t['epochs']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- H200 Benchmarking --- Device: {device}")
    
    # 1. Load Data
    print("Loading and cleaning data...")
    df = load_opra_data(data_path)
    df = clean_and_augment(df)
    dataloader = get_dataloader(df, batch_size=batch_size, shuffle=True)
    
    # 2. Setup Model and Agent
    model = PIKANModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=t['lr'])
    
    # Mock Agent for math throughput testing (dual-process API)
    agent = LiuClawAgent()
    agent.think_fast = lambda step, edge_stats, loss_delta: ReflexDecision(
        reasoning=f"Benchmark step {step}: no action.", prunes=[], lr_adjustment=1.0
    )
    agent.think_slow = lambda history, regime_data, model_state: StrategicDecision(
        reasoning="Benchmark: topology stable.", mutations=[],
        regime_analysis=RegimeThesis(hmm_transition_detected=False),
        training_command="CONTINUE",
    )
    coordinator = EngineCoordinator(agent)
    coordinator.start_threads()
    
    # Heston Params (loaded from config)
    r, kappa, theta, sigma, rho = h['r'], h['kappa'], h['theta'], h['sigma'], h['rho']
    K, T = c['K'], c['T']
    
    # 3. Training Benchmarking
    print(f"Starting Benchmark Training: {epochs} epochs, Batch Size: {batch_size}")
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        running_loss = 0.0
        
        for i, (features, labels) in enumerate(dataloader):
            features, labels = features.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # (S, v, t)
            S, v, t = features[:, 0:1], features[:, 1:2], features[:, 2:3]
            
            # PDE Loss (requires grad on inputs)
            S.requires_grad_(True)
            v.requires_grad_(True)
            t.requires_grad_(True)
            
            pde_loss = heston_pde_loss(model, S, v, t, r, kappa, theta, sigma, rho)
            
            # Total Loss
            total_loss = pde_loss # simplified for benchmark
            total_loss.backward()
            optimizer.step()
            
            # Non-blocking mutation check
            coordinator.apply_pending_mutations(model, optimizer)

            # Occasional agent requests (dual-process)
            if i % 10 == 0:
                coordinator.request_reflex(
                    step=i, edge_stats={}, loss_delta=total_loss.item()
                )
            if i % 50 == 0:
                coordinator.request_strategic(
                    history={"step": i, "loss": total_loss.item()},
                    regime_data={"regime": 0},
                    model_state={"layers": len(model.layers)},
                )
                
            running_loss += total_loss.item()
            
        epoch_end = time.time()
        print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(dataloader):.6f} | Time: {epoch_end-epoch_start:.2f}s")
        
    total_time = time.time() - start_time
    throughput = (len(df) * epochs) / total_time
    print(f"\n--- Benchmark Results ---")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Avg Throughput: {throughput:.2f} samples/sec")
    
    # Save the model
    import os
    if not os.path.exists("models"):
        os.makedirs("models")
    model_path = "models/pikan_heston.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    coordinator.stop_threads()
    return model_path

if __name__ == "__main__":
    benchmark_h200("data/synthetic_opra.csv", epochs=2)
    
