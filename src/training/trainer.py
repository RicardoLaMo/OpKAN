import torch
import torch.nn as nn
import torch.optim as optim
from src.models.heston_pde import heston_pde_loss, heston_boundary_loss

def train_heston_kan(model: nn.Module, 
                     epochs: int,
                     batch_size: int,
                     K: float, r: float, T: float,
                     kappa: float, theta: float, sigma: float, rho: float,
                     lr: float = 1e-3):
    """
    Baseline training loop for the Physics-Informed KAN solving the Heston PDE.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Collocation points for PDE interior
        S_int = torch.rand(batch_size, 1, requires_grad=True) * 200.0  # S in [0, 200]
        v_int = torch.rand(batch_size, 1, requires_grad=True) * 0.5    # v in [0, 0.5]
        t_int = torch.rand(batch_size, 1, requires_grad=True) * T      # t in [0, T]

        # Boundary points: Terminal (t = T)
        S_term = torch.rand(batch_size, 1) * 200.0
        v_term = torch.rand(batch_size, 1) * 0.5
        t_term = torch.ones(batch_size, 1) * T

        # Boundary points: S -> 0
        S_zero = torch.zeros(batch_size, 1)
        v_zero = torch.rand(batch_size, 1) * 0.5
        t_zero = torch.rand(batch_size, 1) * T

        # Boundary points: S -> inf (large S)
        S_inf = torch.ones(batch_size, 1) * 1000.0
        v_inf = torch.rand(batch_size, 1) * 0.5
        t_inf = torch.rand(batch_size, 1) * T

        # Compute Losses
        pde_loss = heston_pde_loss(model, S_int, v_int, t_int, r, kappa, theta, sigma, rho)
        bnd_loss = heston_boundary_loss(model, S_term, v_term, t_term, K,
                                        S_zero, v_zero, t_zero, 
                                        S_inf, v_inf, t_inf, r, T)
        
        # Total Loss
        total_loss = pde_loss + bnd_loss
        
        # Backward and Optimize
        total_loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{epochs} | Total Loss: {total_loss.item():.6f} | PDE Loss: {pde_loss.item():.6f} | Boundary Loss: {bnd_loss.item():.6f}")

    return model
