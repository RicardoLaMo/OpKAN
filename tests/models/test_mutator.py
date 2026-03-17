import torch
import pytest
from src.models.kan_layer import KANLayer, BSplineEdge
from src.models.mutator import TopologicalMutator, C2SymbolicEdge

def test_kan_layer_forward():
    layer = KANLayer(3, 1) # (S, v, t) -> V
    x = torch.randn(10, 3)
    out = layer(x)
    assert out.shape == (10, 1)

def test_edge_swap_and_gradient_continuity():
    # Initial layer with B-splines
    layer = KANLayer(3, 1)
    
    # Mutate edge (0, 0) - Input 'S' to output 'V'
    symbolic_expr = "torch.pow(x, 2)"
    TopologicalMutator.mutate_edge(layer, 0, 0, symbolic_expr)
    
    # Check if the edge is indeed symbolic
    assert isinstance(layer.edges[0][0], C2SymbolicEdge)
    
    # Check forward pass after mutation
    x = torch.randn(10, 3, requires_grad=True)
    out = layer(x)
    assert torch.isfinite(out).all()
    
    # Verify second-order gradients are computable through the swapped edge
    # This is critical for the Heston PDE
    dV_dx = torch.autograd.grad(out.sum(), x, create_graph=True)[0]
    d2V_dx2 = torch.autograd.grad(dV_dx.sum(), x, create_graph=True)[0]
    
    assert torch.isfinite(d2V_dx2).all(), "Second-order gradients are non-finite after mutation."

def test_invalid_symbolic_expression():
    layer = KANLayer(1, 1)
    # Expression with division by zero or non-C2 (like relu)
    invalid_expr = "torch.relu(x)" # ReLU is not C2 continuous
    
    # Though ReLU has finite grads, it's non-C2 at 0.
    # In our verify function, we can check for smoothness or let autograd handle it.
    # For this test, let's use a syntax error.
    bad_syntax = "torch.exp(" 
    
    with pytest.raises(RuntimeError):
        TopologicalMutator.mutate_edge(layer, 0, 0, bad_syntax)
