import numpy as np
import torch
import pytest
from src.data.iv_solver import calculate_iv, black_scholes_price
from src.data.surface import VolatilitySurface

def test_iv_solver_convergence():
    # Known values (S=100, K=100, T=1.0, r=0.05, sigma=0.2)
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
    price = black_scholes_price(S, K, T, r, sigma, 'call')
    
    # Back out IV
    computed_iv = calculate_iv(price, S, K, T, r, 'call')
    
    assert np.isclose(computed_iv, sigma, atol=1e-5), f"IV solver failed. Expected {sigma}, got {computed_iv}"

def test_volatility_surface_continuity():
    # Create synthetic grid for surface fitting
    strikes = np.linspace(80, 120, 10)
    expiries = np.linspace(0.1, 1.0, 5)
    
    K_mesh, T_mesh = np.meshgrid(strikes, expiries)
    K_flat = K_mesh.flatten()
    T_flat = T_mesh.flatten()
    
    # Synthetic IV surface (smile + skew)
    iv_flat = 0.2 + 0.1 * (K_flat - 100)**2 / 1000 + 0.05 * T_flat
    
    surface = VolatilitySurface(K_flat, T_flat, iv_flat)
    
    # Interpolate at point not in grid
    K_test, T_test = 105.0, 0.5
    vol = surface.get_vol(K_test, T_test)
    
    assert np.isfinite(vol), "Surface returned non-finite IV."
    assert 0.1 < vol < 0.5, f"Unreasonable IV returned: {vol}"

def test_black_scholes_put_call_parity():
    # C - P = S - K * exp(-rT)
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
    c = black_scholes_price(S, K, T, r, sigma, 'call')
    p = black_scholes_price(S, K, T, r, sigma, 'put')
    
    lhs = c - p
    rhs = S - K * np.exp(-r * T)
    
    assert np.isclose(lhs, rhs, atol=1e-7), f"Put-Call Parity violated: {lhs} != {rhs}"
