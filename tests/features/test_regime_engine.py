import numpy as np
import pandas as pd
import pytest
from src.features.extractor import extract_regime_features
from src.features.pca_reducer import PCAReducer
from src.models.hmm_regime import RegimeHMM

def test_feature_extraction_shape():
    # Synthetic data
    n = 100
    df = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=n, freq='H'),
        'spot': 100 + np.cumsum(np.random.normal(0, 1, n)),
        'iv': 0.2 + np.abs(np.random.normal(0, 0.05, n))
    })
    
    features = extract_regime_features(df, rolling_window=5)
    
    # Check that output is a dataframe and has expected columns
    assert isinstance(features, pd.DataFrame)
    assert 'log_ret' in features.columns
    assert 'realized_vol' in features.columns
    assert 'iv_rv_spread' in features.columns
    
    # Check shape (n - rolling_window due to shift/rolling)
    assert len(features) <= n

def test_pca_reduction():
    # 10 features, 100 samples
    X = np.random.randn(100, 10)
    reducer = PCAReducer(variance_threshold=0.9)
    X_pca = reducer.fit_transform(X)
    
    assert X_pca.shape[1] < 10, "PCA did not reduce dimensionality."
    assert reducer.get_explained_variance() >= 0.9, "PCA failed to retain target variance."

def test_hmm_regime_prediction():
    # Use a fixed seed and well-separated means so the HMM reliably finds two states.
    rng = np.random.default_rng(42)
    r0 = rng.normal(loc=0.0, scale=0.05, size=(150, 1))  # tight cluster near 0
    r1 = rng.normal(loc=2.0, scale=0.05, size=(150, 1))  # tight cluster near 2

    X = np.vstack([r0, r1])

    hmm = RegimeHMM(n_regimes=2)
    hmm.fit(X)

    preds = hmm.predict_regimes(X)

    # HMM must discover exactly two states
    assert len(np.unique(preds)) == 2, "HMM did not find two distinct states"

    # Each segment should be dominated (>90%) by its own majority label
    maj_first  = np.bincount(preds[:150]).argmax()
    maj_second = np.bincount(preds[150:]).argmax()
    assert np.mean(preds[:150]  == maj_first)  > 0.9, "First segment not coherent"
    assert np.mean(preds[150:]  == maj_second) > 0.9, "Second segment not coherent"
    assert maj_first != maj_second, "Both segments assigned the same regime label"
