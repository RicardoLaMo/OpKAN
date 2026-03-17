import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class BSplineEdge(nn.Module):
    """
    A proper B-spline activation edge for KAN layers.

    Each edge computes: output = w_base * silu(x) + w_spline * spline(x)
    where spline(x) = sum_i c_i * B_{i,k}(x) with cubic (k=3) B-spline basis.

    This replaces the placeholder tanh implementation with the architecture
    described in Liu et al. (2024) "KAN: Kolmogorov-Arnold Networks".
    """

    def __init__(self, grid_size: int = 5, spline_order: int = 3):
        super().__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        # Build extended grid on [-1, 1] with (spline_order) padding knots on each side
        h = 2.0 / grid_size
        grid = torch.arange(-(spline_order), grid_size + spline_order + 1, dtype=torch.float32) * h - 1.0
        self.register_buffer('grid', grid)  # shape: (grid_size + 2*spline_order + 1,)

        # Learnable spline coefficients; n_basis = grid_size + spline_order
        n_basis = grid_size + spline_order
        self.coefficients = nn.Parameter(torch.zeros(n_basis))
        nn.init.normal_(self.coefficients, std=0.1)

        # Residual base weights (from KAN paper: output = w_b*b(x) + w_s*spline(x))
        self.w_base = nn.Parameter(torch.ones(1))
        self.w_spline = nn.Parameter(torch.ones(1))

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute B-spline basis values via Cox-de Boor recursion.

        Args:
            x: (batch, 1) input tensor, clamped to grid domain.
        Returns:
            bases: (batch, n_basis) basis values.
        """
        grid = self.grid  # (G + 2k + 1,)
        x = x.clamp(self.grid[0], self.grid[-1])  # clamp to domain

        # Order-0 indicator basis: (batch, G+2k) intervals
        # bases[b, i] = 1 if grid[i] <= x[b] < grid[i+1]
        bases = ((x >= grid[:-1]) & (x < grid[1:])).float()  # (batch, G+2k)

        # Handle right boundary: include x == grid[-1] in last interval
        bases[:, -1] = bases[:, -1] + (x[:, 0] == grid[-1]).float()

        # Cox-de Boor recursion: k steps from order-0 to order-spline_order.
        # At each step k, we go from m basis functions to m-1.
        # For output index i in [0, m-2]:
        #   B[i,k](x) = (x-t[i])/(t[i+k]-t[i]) * B[i,k-1](x)
        #             + (t[i+k+1]-x)/(t[i+k+1]-t[i+1]) * B[i+1,k-1](x)
        for k in range(1, self.spline_order + 1):
            m = bases.shape[1]  # current number of basis functions

            # Left weight: (x - t[i]) / (t[i+k] - t[i])  for i = 0..m-2
            t_i = grid[:m - 1].unsqueeze(0)       # (1, m-1)
            t_ik = grid[k:m - 1 + k].unsqueeze(0)  # (1, m-1)
            denom_left = (t_ik - t_i).clamp(min=1e-8)
            left = (x - t_i) / denom_left          # (batch, m-1)

            # Right weight: (t[i+k+1] - x) / (t[i+k+1] - t[i+1])  for i = 0..m-2
            t_i1 = grid[1:m].unsqueeze(0)              # (1, m-1)
            t_ik1 = grid[k + 1:m - 1 + k + 1].unsqueeze(0)  # (1, m-1)
            denom_right = (t_ik1 - t_i1).clamp(min=1e-8)
            right = (t_ik1 - x) / denom_right          # (batch, m-1)

            # New basis: B[i,k] = left[i]*B[i,k-1] + right[i]*B[i+1,k-1]
            bases = left * bases[:, :-1] + right * bases[:, 1:]
            # shape: (batch, m-1)

        return bases  # (batch, grid_size + spline_order)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 1)
        Returns:
            out: (batch, 1)
        """
        bases = self.b_splines(x)                         # (batch, n_basis)
        spline_out = (bases * self.coefficients).sum(-1, keepdim=True)  # (batch, 1)
        base_out = F.silu(x)                              # (batch, 1)
        return self.w_base * base_out + self.w_spline * spline_out


class KANLayer(nn.Module):
    """
    A KAN Layer where each edge is an independent B-spline activation module.
    Each edge connects one input neuron to one output neuron.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.edges = nn.ModuleList([
            nn.ModuleList([BSplineEdge() for _ in range(out_features)])
            for _ in range(in_features)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_features)
        batch_size = x.shape[0]
        output = torch.zeros(batch_size, self.out_features, device=x.device)

        for i in range(self.in_features):
            for j in range(self.out_features):
                output[:, j] += self.edges[i][j](x[:, i:i+1]).squeeze(-1)

        return output

    def swap_edge(self, in_idx: int, out_idx: int, new_module: nn.Module):
        """Physically swap an edge module and move to correct device."""
        device = next(self.parameters()).device
        self.edges[in_idx][out_idx] = new_module.to(device)
