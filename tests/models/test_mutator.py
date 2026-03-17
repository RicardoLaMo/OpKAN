import torch
import pytest
from src.models.kan_layer import KANLayer, BSplineEdge
from src.models.mutator import TopologicalMutator, C2SymbolicEdge


class PIKANModel(torch.nn.Module):
    """Minimal 2-layer model matching production layout."""
    def __init__(self, layer_dims):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            KANLayer(layer_dims[i], layer_dims[i + 1])
            for i in range(len(layer_dims) - 1)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def test_kan_layer_forward():
    layer = KANLayer(3, 1)  # (S, v, t) -> V
    x = torch.randn(10, 3)
    out = layer(x)
    assert out.shape == (10, 1)


def test_edge_swap_and_gradient_continuity():
    model = PIKANModel([3, 16, 1])

    # Mutate edge from input 'S' (node 0) to first hidden node (node 0 in layer 1)
    symbolic_expr = "torch.pow(x, 2)"
    TopologicalMutator.mutate_edge(model, "L0_N0_to_L1_N0", "REPLACE", symbolic_expr)

    # The replaced edge should be a C2SymbolicEdge
    assert isinstance(model.layers[0].edges[0][0], C2SymbolicEdge)

    # Forward pass must still work
    x = torch.randn(10, 3, requires_grad=True)
    out = model(x)
    assert torch.isfinite(out).all()

    # Second-order gradients must be computable (required for Heston PDE)
    dV_dx = torch.autograd.grad(out.sum(), x, create_graph=True)[0]
    d2V_dx2 = torch.autograd.grad(dV_dx.sum(), x, create_graph=True)[0]
    assert torch.isfinite(d2V_dx2).all(), "Second-order gradients non-finite after mutation."


def test_invalid_symbolic_expression():
    model = PIKANModel([3, 1])

    # Syntax error in formula should raise ValueError or RuntimeError
    bad_syntax = "torch.exp("
    with pytest.raises((ValueError, RuntimeError)):
        TopologicalMutator.mutate_edge(model, "L0_N0_to_L1_N0", "REPLACE", bad_syntax)

def test_security_ast_validation():
    """Verifies that code injection attempts are blocked by the AST allowlist."""
    model = PIKANModel([3, 1])
    
    # 1. Block os.system injection
    injection = "__import__('os').system('echo INJECTED')"
    with pytest.raises(ValueError, match="Only 'torch.*' attributes are allowed|Disallowed name"):
        TopologicalMutator.mutate_edge(model, "L0_N0_to_L1_N0", "REPLACE", injection)
        
    # 2. Block non-allowlisted torch functions
    unsafe_torch = "torch.save(x, 'model.pt')"
    with pytest.raises(ValueError, match="not in the allowlist"):
        TopologicalMutator.mutate_edge(model, "L0_N0_to_L1_N0", "REPLACE", unsafe_torch)

def test_prune_zero_edge():
    """Verifies that PRUNE results in a ZeroEdge that outputs actual zeros."""
    from src.models.mutator import ZeroEdge
    model = PIKANModel([3, 1])
    
    TopologicalMutator.mutate_edge(model, "L0_N0_to_L1_N0", "PRUNE")
    
    edge = model.layers[0].edges[0][0]
    assert isinstance(edge, ZeroEdge)
    
    x = torch.randn(5, 1)
    out = edge(x)
    assert (out == 0).all()

def test_atomic_rollback():
    """Verifies that a failed mutation batch is completely rolled back."""
    model = PIKANModel([3, 1])
    
    # Save original edge
    original_edge = model.layers[0].edges[0][0]
    
    # Simulate a partial failure: 
    # 1st mutation is valid, 2nd mutation is invalid (security violation)
    from src.engine.coordinator import EngineCoordinator
    from src.agent.core import LiuClawAgent
    from src.agent.dsl import StrategicDecision, EdgeMutation, RegimeThesis
    
    decision = StrategicDecision(
        reasoning="Testing rollback",
        mutations=[
            EdgeMutation(edge_id="L0_N0_to_L1_N0", action="REPLACE", formula="torch.sin(x)", reasoning="R1"),
            EdgeMutation(edge_id="L0_N1_to_L1_N0", action="REPLACE", formula="__import__('os').system('fail')", reasoning="R2")
        ],
        regime_analysis=RegimeThesis(hmm_transition_detected=False)
    )
    
    # Attempt to apply via coordinator (which implements rollback logic)
    from src.engine.queues import strategic_decision_queue
    strategic_decision_queue.put(decision)
    
    coordinator = EngineCoordinator(agent=LiuClawAgent())
    coordinator.apply_pending_mutations(model)
    
    # Verify rollback: L0_N0_to_L1_N0 should be a BSplineEdge (rolled back), NOT torch.sin(x)
    edge = model.layers[0].edges[0][0]
    assert isinstance(edge, BSplineEdge)
    assert not isinstance(edge, C2SymbolicEdge)



