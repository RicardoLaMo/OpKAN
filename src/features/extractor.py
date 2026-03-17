import pandas as pd
import numpy as np

def extract_regime_features(df: pd.DataFrame, rolling_window: int = 20) -> pd.DataFrame:
    """
    Extracts time-series features for market regime detection.
    Features: log-returns, realized volatility, IV-RV spread.
    """
    # Ensure dataframe is sorted by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # 1. Log-returns of the spot price
    df['log_ret'] = np.log(df['spot'] / df['spot'].shift(1))
    
    # 2. Realized Volatility (annualized)
    df['realized_vol'] = df['log_ret'].rolling(window=rolling_window).std() * np.sqrt(252 * 6.5 * 60) # Scaled for intraday if needed
    
    # 3. IV Spread (Average IV from options - Realized Vol)
    # Assumes df already has an 'iv' column from the data pipeline
    df['iv_rv_spread'] = df['iv'] - df['realized_vol']
    
    # 4. IV Momentum
    df['iv_mom'] = df['iv'].diff()
    
    # Drop rows with NaN from rolling calculations
    features_df = df[['log_ret', 'realized_vol', 'iv', 'iv_rv_spread', 'iv_mom']].dropna()
    
    return features_df
