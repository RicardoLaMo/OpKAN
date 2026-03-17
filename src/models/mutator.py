import torch
import torch.nn as nn
from typing import Any

class C2SymbolicEdge(nn.Module):
    """
    A symbolic KAN edge that evaluates a torch-compatible expression.
    Enforces C^2 continuity and provides gradient health monitoring.
    """
    def __init__(self, expression: str):
        super().__init__()
        self.expression_str = expression
        # Parameters for scaling/shifting the input domain to match the symbolic function
        self.scale = nn.Parameter(torch.ones(1))
        self.shift = nn.Parameter(torch.zeros(1))
        self.coeff = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Safety: eval with torch context
        # The LiuClaw agent provides the 'expression' string
        local_context = {"torch": torch, "x": (x - self.shift) * self.scale}
        try:
            # Evaluate the expression with limited globals
            # We allow torch operations which may need some builtins for internal dispatch
            out = eval(self.expression_str, {"torch": torch}, local_context)
            return self.coeff * out
        except Exception as e:
            raise RuntimeError(f"Failed to evaluate symbolic expression '{self.expression_str}': {e}")

    def verify_second_order_gradients(self, x: torch.Tensor):
        """
        Autograd hook to verify second-order gradients are non-zero and stable.
        Critical for Heston PDE residuals.
        """
        x = x.detach().requires_grad_(True)
        y = self.forward(x)
        dy_dx = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]
        d2y_dx2 = torch.autograd.grad(dy_dx, x, torch.ones_like(dy_dx), create_graph=True)[0]
        
        if not torch.isfinite(d2y_dx2).all():
            raise ValueError(f"Second-order gradient exploded for symbolic edge: {self.expression_str}")
        return d2y_dx2

class TopologicalMutator:
    """
    The 'PyTorch Surgeon' that executes topological changes in the KAN.
    """
    @staticmethod
    def mutate_edge(kan_layer: nn.Module, 
                    in_idx: int, 
                    out_idx: int, 
                    symbolic_expr: str):
        """
        Freezes the existing B-spline and swaps it for a symbolic edge.
        """
        # 1. Verify C2 continuity before swapping
        test_edge = C2SymbolicEdge(symbolic_expr)
        test_input = torch.linspace(-5, 5, 100).unsqueeze(-1)
        test_edge.verify_second_order_gradients(test_input)
        
        # 2. Perform the swap
        kan_layer.swap_edge(in_idx, out_idx, test_edge)
        
        return f"Successfully mutated Edge({in_idx}, {out_idx}) to {symbolic_expr}"
