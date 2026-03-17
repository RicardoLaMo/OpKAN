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
from unittest.mock import MagicMock
from src.agent.dsl import LiuClawDecision

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

def benchmark_h200(data_path: str, batch_size: int = 1024, epochs: int = 10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- H200 Benchmarking --- Device: {device}")
    
    # 1. Load Data
    print("Loading and cleaning data...")
    df = load_opra_data(data_path)
    df = clean_and_augment(df)
    dataloader = get_dataloader(df, batch_size=batch_size, shuffle=True)
    
    # 2. Setup Model and Agent
    model = PIKANModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Mock Agent for math throughput testing
    agent = MagicMock(spec=LiuClawAgent)
    agent.decide_mutations.return_value = LiuClawDecision(reasoning="Mock", confidence=1.0)
    coordinator = EngineCoordinator(agent)
    coordinator.start_agent_thread()
    
    # Heston Params
    r, kappa, theta, sigma, rho, K, T = 0.05, 2.0, 0.04, 0.3, -0.7, 100.0, 1.0
    
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
            coordinator.apply_pending_mutations(model)
            
            # Occasional agent request
            if i % 50 == 0:
                coordinator.request_mutation({"step": i}, {"health": "benchmark"})
                
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
    
    coordinator.stop_agent_thread()
    return model_path

if __name__ == "__main__":
    benchmark_h200("data/synthetic_opra.csv", epochs=2)
    
