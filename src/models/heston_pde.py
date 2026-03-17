import torch
import torch.nn as nn

def heston_pde_loss(model: nn.Module, 
                    S: torch.Tensor, 
                    v: torch.Tensor, 
                    t: torch.Tensor, 
                    r: float, 
                    kappa: float, 
                    theta: float, 
                    sigma: float, 
                    rho: float) -> torch.Tensor:
    """
    Computes the Heston PDE residual.
    Requires input tensors to have requires_grad=True.
    """
    if not S.requires_grad:
        S.requires_grad_(True)
    if not v.requires_grad:
        v.requires_grad_(True)
    if not t.requires_grad:
        t.requires_grad_(True)
        
    # Forward pass
    V = model(torch.cat([S, v, t], dim=1))
    
    # First derivatives
    dV_dS = torch.autograd.grad(V, S, grad_outputs=torch.ones_like(V), create_graph=True, retain_graph=True)[0]
    dV_dv = torch.autograd.grad(V, v, grad_outputs=torch.ones_like(V), create_graph=True, retain_graph=True)[0]
    dV_dt = torch.autograd.grad(V, t, grad_outputs=torch.ones_like(V), create_graph=True, retain_graph=True)[0]
    
    # Second derivatives
    d2V_dS2 = torch.autograd.grad(dV_dS, S, grad_outputs=torch.ones_like(dV_dS), create_graph=True, retain_graph=True)[0]
    d2V_dv2 = torch.autograd.grad(dV_dv, v, grad_outputs=torch.ones_like(dV_dv), create_graph=True, retain_graph=True)[0]
    d2V_dSdv = torch.autograd.grad(dV_dS, v, grad_outputs=torch.ones_like(dV_dS), create_graph=True, retain_graph=True)[0]
    
    # Heston PDE Residual
    residual = (dV_dt 
              + 0.5 * v * S**2 * d2V_dS2 
              + rho * sigma * v * S * d2V_dSdv 
              + 0.5 * sigma**2 * v * d2V_dv2 
              + r * S * dV_dS 
              + kappa * (theta - v) * dV_dv 
              - r * V)
              
    return torch.mean(residual**2)


def heston_boundary_loss(model: nn.Module, 
                         S_term: torch.Tensor, v_term: torch.Tensor, t_term: torch.Tensor, K: float,
                         S_zero: torch.Tensor, v_zero: torch.Tensor, t_zero: torch.Tensor,
                         S_inf: torch.Tensor, v_inf: torch.Tensor, t_inf: torch.Tensor, 
                         r: float, T: float) -> torch.Tensor:
    """
    Computes the boundary conditions loss.
    """
    # 1. Terminal condition: V(S, v, T) = max(S - K, 0)
    V_term_pred = model(torch.cat([S_term, v_term, t_term], dim=1))
    V_term_true = torch.clamp(S_term - K, min=0.0)
    loss_term = torch.mean((V_term_pred - V_term_true)**2)

    # 2. S -> 0 condition: V(0, v, t) = 0
    V_zero_pred = model(torch.cat([S_zero, v_zero, t_zero], dim=1))
    loss_zero = torch.mean((V_zero_pred - 0.0)**2)

    # 3. S -> inf condition: V(S, v, t) ≈ S - K * exp(-r * (T - t))
    V_inf_pred = model(torch.cat([S_inf, v_inf, t_inf], dim=1))
    V_inf_true = S_inf - K * torch.exp(-r * (T - t_inf))
    loss_inf = torch.mean((V_inf_pred - V_inf_true)**2)

    return loss_term + loss_zero + loss_inf
