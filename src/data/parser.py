import pandas as pd
import numpy as np

def load_opra_data(file_path: str) -> pd.DataFrame:
    """
    Parses Databento/OPRA-style options data and performs cleaning.
    """
    df = pd.read_csv(file_path)
    
    # 1. Standardize columns (assume common names from Databento)
    # Mapping might be needed depending on the exact schema
    mapping = {
        'bid_px': 'bid',
        'ask_px': 'ask',
        'strike_px': 'strike',
        'expiration': 'expiry',
        'und_px': 'spot',
        'ts_recv': 'timestamp'
    }
    df = df.rename(columns=mapping)
    
    # 2. Filter zero bids and asks
    df = df[(df['bid'] > 0) & (df['ask'] > 0)]
    
    # 3. Calculate mid-price
    df['price'] = (df['bid'] + df['ask']) / 2.0
    
    # 4. Handle time-to-maturity (T) in years
    # Assume expiry and timestamp are datetime-like
    df['expiry'] = pd.to_datetime(df['expiry'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['T'] = (df['expiry'] - df['timestamp']).dt.total_seconds() / (365.0 * 24.0 * 3600.0)
    
    # 5. Filter out expired or stale options
    df = df[df['T'] > (1/365.0)] # Must have at least 1 day remaining
    
    return df

def clean_and_augment(df: pd.DataFrame, r: float = 0.05) -> pd.DataFrame:
    """
    Calculates IV and filters NaN results.
    """
    from src.data.iv_solver import calculate_iv_batch
    
    # Identify calls and puts
    # Assumes 'side' or 'type' column exists (C/P)
    option_types = df['type'].map({'C': 'call', 'P': 'put'}).values
    
    # Batch calculate IV
    ivs = calculate_iv_batch(
        df['price'].values, 
        df['spot'].values, 
        df['strike'].values, 
        df['T'].values, 
        r, 
        option_types
    )
    
    df['iv'] = ivs
    
    # Filter out records where IV calculation failed
    df = df.dropna(subset=['iv'])
    
    return df
