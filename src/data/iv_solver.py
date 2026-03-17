import torch
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

def black_scholes_price(S: float, K: float, T: float, r: float, sigma: float, option_type: str = 'call') -> float:
    """Standard Black-Scholes-Merton European Option Pricing."""
    if T <= 0:
        return max(S - K, 0.0) if option_type == 'call' else max(K - S, 0.0)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price

def calculate_iv(price: float, S: float, K: float, T: float, r: float, option_type: str = 'call') -> float:
    """Calculate Implied Volatility via Brent's method."""
    if T <= 0:
        return 0.0
    
    # Objective function: price_error(sigma) = BS(sigma) - target_price
    objective = lambda sigma: black_scholes_price(S, K, T, r, sigma, option_type) - price
    
    try:
        # Initial search interval [1e-4, 5.0] (0.01% to 500% IV)
        return brentq(objective, 1e-4, 5.0, xtol=1e-6)
    except (ValueError, RuntimeError):
        # Fallback or error
        return np.nan

def calculate_iv_batch(prices: np.ndarray, S: np.ndarray, K: np.ndarray, T: np.ndarray, r: float, option_types: np.ndarray) -> np.ndarray:
    """Batch calculation of IVs (useful for preprocessing)."""
    ivs = []
    for p, s, k, t, ot in zip(prices, S, K, T, option_types):
        ivs.append(calculate_iv(p, s, k, t, r, ot))
    return np.array(ivs)
