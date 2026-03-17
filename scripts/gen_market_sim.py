import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_high_fidelity_opra(n_rows: int = 1000000, out_path: str = "data/real_market_sim.csv"):
    """
    Generates a high-fidelity synthetic dataset with distinct market regimes.
    Regime 0: Low Vol (Diffusion)
    Regime 1: High Vol (Expansion)
    Regime 2: Jump/Crash
    """
    print(f"🚀 Generating {n_rows} rows of High-Fidelity Market Data...")
    
    S0 = 100.0
    start_ts = datetime(2023, 1, 1, 9, 30)
    ts = [start_ts + timedelta(milliseconds=i*10) for i in range(n_rows)]
    
    # Regime Windows
    diffusion_end = int(n_rows * 0.4)
    expansion_end = int(n_rows * 0.8)
    
    # Returns with regime-switching volatility
    vol = np.zeros(n_rows)
    vol[:diffusion_end] = 0.15 / np.sqrt(252 * 6.5 * 3600 * 100) # Low Vol
    vol[diffusion_end:expansion_end] = 0.45 / np.sqrt(252 * 6.5 * 3600 * 100) # High Vol
    vol[expansion_end:] = 0.80 / np.sqrt(252 * 6.5 * 3600 * 100) # Crash Vol
    
    returns = np.random.normal(0, vol)
    # Add jumps to the crash regime
    jumps = np.random.choice([0, -0.02, -0.05], size=n_rows, p=[0.99, 0.007, 0.003])
    returns[expansion_end:] += jumps[expansion_end:]
    
    spot = S0 * np.exp(np.cumsum(returns))
    strikes = np.round(spot + np.random.normal(0, 15, n_rows), 2)
    expiries = [t + timedelta(days=np.random.randint(7, 180)) for t in ts]
    types = np.random.choice(['C', 'P'], n_rows)
    
    T = np.array([(e - t).total_seconds() / (365*24*3600) for e, t in zip(expiries, ts)])
    # Market IV (Skewed)
    iv = np.zeros(n_rows)
    iv[:diffusion_end] = 0.2
    iv[diffusion_end:expansion_end] = 0.4
    iv[expansion_end:] = 0.6
    # Add skew: OTM Puts have higher IV
    moneyness = strikes / spot
    iv += 0.1 * np.maximum(1.0 - moneyness, 0) # Put skew
    
    # Black-Scholes Price (approximation for label)
    from src.data.iv_solver import black_scholes_price
    # Vectorized BS is faster but for simplicity we'll use a heuristic for 1M rows
    # or just use a small subset for the pricer and interpolate. 
    # For the stress test, a heuristic is fine.
    intrinsic = np.where(types == 'C', np.maximum(spot - strikes, 0), np.maximum(strikes - spot, 0))
    # Approximation: V = Intrinsic + 0.4 * S * IV * sqrt(T)
    prices = intrinsic + 0.4 * spot * iv * np.sqrt(T) + np.random.normal(0, 0.01, n_rows)
    prices = np.maximum(prices, 0.01)
    
    df = pd.DataFrame({
        'ts_recv': ts,
        'und_px': spot,
        'strike_px': strikes,
        'bid_px': prices - 0.02,
        'ask_px': prices + 0.02,
        'expiration': expiries,
        'type': types
    })
    
    df.to_csv(out_path, index=False)
    print(f"✅ High-Fidelity Dataset saved to {out_path} ({len(df)} rows)")

if __name__ == "__main__":
    generate_high_fidelity_opra(n_rows=1000000)
