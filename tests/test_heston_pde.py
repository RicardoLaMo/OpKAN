import torch
import torch.nn as nn
from src.models.heston_pde import heston_pde_loss, heston_boundary_loss

class DummyKAN(nn.Module):
    # simple neural net to mock the KAN for testing gradients
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 16),
            nn.Tanh(),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        return self.net(x)

def test_heston_pde_loss_autograd():
    model = DummyKAN()
    
    # Collocation points
    batch_size = 100
    S = torch.rand(batch_size, 1, requires_grad=True) * 100 + 50
    v = torch.rand(batch_size, 1, requires_grad=True) * 0.1
    t = torch.rand(batch_size, 1, requires_grad=True) * 1.0
    
    r, kappa, theta, sigma, rho = 0.05, 2.0, 0.04, 0.1, -0.7
    
    loss = heston_pde_loss(model, S, v, t, r, kappa, theta, sigma, rho)
    
    # Assert loss is computed
    assert torch.isfinite(loss), "Loss is not finite."
    
    # Assert gradients can flow back to model parameters
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Gradient not computed for {name}"
        assert torch.isfinite(param.grad).all(), f"Gradient exploding/nan for {name}"

def test_heston_boundary_loss():
    model = DummyKAN()
    batch_size = 50
    K = 100.0
    r = 0.05
    T = 1.0

    S_term = torch.rand(batch_size, 1) * 200
    v_term = torch.rand(batch_size, 1) * 0.1
    t_term = torch.ones(batch_size, 1) * T

    S_zero = torch.zeros(batch_size, 1)
    v_zero = torch.rand(batch_size, 1) * 0.1
    t_zero = torch.rand(batch_size, 1) * T

    S_inf = torch.ones(batch_size, 1) * 1000
    v_inf = torch.rand(batch_size, 1) * 0.1
    t_inf = torch.rand(batch_size, 1) * T

    loss = heston_boundary_loss(model, S_term, v_term, t_term, K,
                                S_zero, v_zero, t_zero, S_inf, v_inf, t_inf, r, T)
    
    assert torch.isfinite(loss), "Boundary loss is not finite."
