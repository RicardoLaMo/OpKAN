import numpy as np
import pandas as pd
from scipy.interpolate import SmoothBivariateSpline

class VolatilitySurface:
    """
    Fits and provides a 2D cubic spline interpolation for the volatility surface.
    Coordinates: Strike (K), Time-to-Maturity (T) -> Implied Volatility (v)
    """
    def __init__(self, K: np.ndarray, T: np.ndarray, iv: np.ndarray):
        # Fit a smooth bivariate spline
        # kx, ky = 3 for cubic spline (C2 continuous)
        # s is the smoothing factor
        self.spline = SmoothBivariateSpline(K, T, iv, kx=3, ky=3, s=len(iv)*0.01)
    
    def get_vol(self, K: float | np.ndarray, T: float | np.ndarray) -> float | np.ndarray:
        """Query the surface for IV."""
        return self.spline.ev(K, T)

def fit_surface(df: pd.DataFrame) -> VolatilitySurface:
    """Helper to fit surface from cleaned DataFrame."""
    return VolatilitySurface(
        df['strike'].values, 
        df['T'].values, 
        df['iv'].values
    )
