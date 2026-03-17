import torch
import torch.nn as nn
from typing import Any, Tuple


class ZeroEdge(nn.Module):
    """Edge that always outputs zeros, used by the PRUNE action."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)


class C2SymbolicEdge(nn.Module):
    """
    A symbolic KAN edge that evaluates a torch-compatible expression.
    Enforces C^2 continuity and provides gradient health monitoring.
    """
    def __init__(self, expression: str):
        super().__init__()
        self.expression_str = expression
        self.scale = nn.Parameter(torch.ones(1))
        self.shift = nn.Parameter(torch.zeros(1))
        self.coeff = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local_context = {"torch": torch, "x": (x - self.shift) * self.scale}
        try:
            out = eval(self.expression_str, {"torch": torch}, local_context)
            return self.coeff * out
        except Exception as e:
            raise RuntimeError(f"Failed to evaluate symbolic expression '{self.expression_str}': {e}")

    def verify_second_order_gradients(self, x: torch.Tensor):
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
    Supports PRUNE, REPLACE, and KEEP actions.
    """
    @staticmethod
    def parse_edge_id(edge_id: str) -> Tuple[int, int, int]:
        """
        Parses 'L0_N0_to_L1_N1' into (layer_idx, in_idx, out_idx).
        Note: The user provided 'L1_N0_to_L2_N1', we assume L0 is the first layer.
        We'll parse L{i}_N{j}_to_L{k}_N{m}.
        """
        try:
            parts = edge_id.split('_')
            layer_idx = int(parts[0][1:])
            in_idx = int(parts[1][1:])
            out_idx = int(parts[4][1:])
            return layer_idx, in_idx, out_idx
        except Exception as e:
            raise ValueError(f"Invalid edge_id format '{edge_id}': {e}")

    @staticmethod
    def mutate_edge(model: nn.Module, 
                    edge_id: str, 
                    action: str, 
                    formula: str = None,
                    initial_params: dict = None):
        """
        Executes a mutation command on the KAN model.
        """
        layer_idx, in_idx, out_idx = TopologicalMutator.parse_edge_id(edge_id)
        target_layer = model.layers[layer_idx]

        if action == "KEEP":
            return f"Edge({edge_id}) kept as is."

        if action == "PRUNE":
            target_layer.swap_edge(in_idx, out_idx, ZeroEdge())
            return f"Edge({edge_id}) pruned."

        if action == "REPLACE":
            if not formula:
                raise ValueError(f"Formula required for REPLACE action on {edge_id}")
            
            # 1. Create and initialize the symbolic edge
            new_edge = C2SymbolicEdge(formula)
            if initial_params:
                with torch.no_grad():
                    if 'scale' in initial_params: new_edge.scale.fill_(initial_params['scale'])
                    if 'shift' in initial_params: new_edge.shift.fill_(initial_params['shift'])
                    if 'coeff' in initial_params: new_edge.coeff.fill_(initial_params['coeff'])

            # 2. Verify C2 continuity
            test_input = torch.linspace(-5, 5, 100).unsqueeze(-1)
            new_edge.verify_second_order_gradients(test_input)
            
            # 3. Perform swap
            target_layer.swap_edge(in_idx, out_idx, new_edge)
            return f"Edge({edge_id}) mutated to {formula}"

        return f"Unknown action {action} for {edge_id}"
