import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.data.parser import load_opra_data, clean_and_augment
from src.data.dataset import get_dataloader
from src.models.kan_layer import KANLayer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

def run_backtest(data_path: str, model_path: str = "models/pikan_heston.pt"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- PI-KAN Backtesting --- Device: {device}")
    
    # 1. Load Data
    df = load_opra_data(data_path)
    df = clean_and_augment(df)
    
    # Split: 80% Train / 20% Test (Backtest)
    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:].copy()
    
    # 2. Load Model
    model = PIKANModel().to(device)
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    # 3. Predict
    S = torch.tensor(test_df['spot'].values, dtype=torch.float32).to(device)
    v = torch.tensor(test_df['iv'].values, dtype=torch.float32).to(device)
    t = torch.tensor(test_df['T'].values, dtype=torch.float32).to(device)
    features = torch.stack([S, v, t], dim=1)
    
    with torch.no_grad():
        V_pred = model(features).cpu().numpy().flatten()
    
    V_true = test_df['price'].values
    
    # 4. Metrics
    mae = mean_absolute_error(V_true, V_pred)
    rmse = np.sqrt(mean_squared_error(V_true, V_pred))
    r2 = r2_score(V_true, V_pred)
    
    print(f"\n--- Backtest Results ---")
    print(f"MAE: {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"R^2 Score: {r2:.6f}")
    
    # 5. Simple Visualization
    plt.figure(figsize=(10, 6))
    plt.scatter(V_true, V_pred, alpha=0.5, s=2)
    plt.plot([V_true.min(), V_true.max()], [V_true.min(), V_true.max()], 'r--')
    plt.xlabel("Market Price (Synthetic)")
    plt.ylabel("PI-KAN Predicted Price")
    plt.title("PI-KAN vs Market Reality (Heston Dynamics)")
    plt.savefig("backtest_results.png")
    print("Saved backtest plot to backtest_results.png")

if __name__ == "__main__":
    import os
    from scripts.benchmark_h200 import benchmark_h200
    
    data_path = "data/synthetic_opra.csv"
    model_path = "models/pikan_heston.pt"
    
    # 1. Run a slightly longer training to get a decent model for backtesting
    # In a real environment, we'd use more epochs
    print("Pre-training PI-KAN for backtest...")
    benchmark_h200(data_path, epochs=3, batch_size=1024)
    
    # 2. Run Backtest
    run_backtest(data_path, model_path)
