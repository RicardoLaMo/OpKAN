import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_synthetic_opra(n_rows: int = 1000000, out_path: str = "data/synthetic_opra.csv"):
    """
    Generates a large-scale synthetic dataset mimicking Databento/OPRA options data.
    Columns: ts_recv, und_px, strike_px, bid_px, ask_px, expiration, type
    """
    print(f"Generating {n_rows} rows of synthetic OPRA data...")
    
    # Base parameters
    S0 = 100.0
    start_ts = datetime(2023, 1, 1, 9, 30)
    
    # 1. Timestamps (intraday)
    ts = [start_ts + timedelta(milliseconds=i*10) for i in range(n_rows)]
    
    # 2. Spot price (Geometric Brownian Motion)
    returns = np.random.normal(0, 0.0001, n_rows)
    spot = S0 * np.exp(np.cumsum(returns))
    
    # 3. Strikes (clustered around spot)
    strikes = np.round(spot + np.random.normal(0, 10, n_rows), 2)
    
    # 4. Expirations (1 month to 1 year)
    expiries = [t + timedelta(days=np.random.randint(30, 365)) for t in ts]
    
    # 5. Type (Call/Put)
    types = np.random.choice(['C', 'P'], n_rows)
    
    # 6. Prices (Heuristic)
    # Simple intrinsic + time value approximation
    T = np.array([(e - t).total_seconds() / (365*24*3600) for e, t in zip(expiries, ts)])
    intrinsic = np.where(types == 'C', np.maximum(spot - strikes, 0), np.maximum(strikes - spot, 0))
    time_value = 5.0 * np.sqrt(T) * (spot / 100.0)
    mid_price = intrinsic + time_value + np.random.normal(0, 0.1, n_rows)
    mid_price = np.maximum(mid_price, 0.05)
    
    bid = mid_price - 0.05
    ask = mid_price + 0.05
    
    df = pd.DataFrame({
        'ts_recv': ts,
        'und_px': spot,
        'strike_px': strikes,
        'bid_px': bid,
        'ask_px': ask,
        'expiration': expiries,
        'type': types
    })
    
    df.to_csv(out_path, index=False)
    print(f"Dataset saved to {out_path}")

if __name__ == "__main__":
    # Generate a medium-sized dataset for testing/benchmarking
    generate_synthetic_opra(n_rows=100000)
