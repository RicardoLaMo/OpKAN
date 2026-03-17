import torch
import torch.nn as nn
from typing import Callable, List

class BSplineEdge(nn.Module):
    """
    Mock B-spline edge for the KAN.
    Simplified as a trainable Tanh for the demo, but swap-ready.
    """
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1))
        self.bias = nn.Parameter(torch.randn(1))
        
    def forward(self, x):
        return self.weight * torch.tanh(x) + self.bias

class KANLayer(nn.Module):
    """
    A basic KAN Layer where each edge is an independent module.
    Each edge connects one input neuron to one output neuron.
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Grid of activation functions (edges)
        # Layer structure: [in_features x out_features] edges
        self.edges = nn.ModuleList([
            nn.ModuleList([BSplineEdge() for _ in range(out_features)]) 
            for _ in range(in_features)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, in_features]
        batch_size = x.shape[0]
        output = torch.zeros(batch_size, self.out_features, device=x.device)
        
        for i in range(self.in_features):
            for j in range(self.out_features):
                output[:, j] += self.edges[i][j](x[:, i:i+1]).squeeze(-1)
                
        return output

    def swap_edge(self, in_idx: int, out_idx: int, new_module: nn.Module):
        """Physically swap an edge module."""
        self.edges[in_idx][out_idx] = new_module
