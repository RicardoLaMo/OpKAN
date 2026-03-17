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
    # Create two distinct regimes
    # Regime 0: Low vol, Regime 1: High vol
    r0 = np.random.normal(0, 0.1, (100, 1))
    r1 = np.random.normal(0, 1.0, (100, 1))
    X = np.vstack([r0, r1])
    
    hmm = RegimeHMM(n_regimes=2)
    hmm.fit(X)
    
    preds = hmm.predict_regimes(X)
    
    # Check if HMM identifies two distinct states roughly aligned with the split
    assert len(np.unique(preds)) == 2
    # Check roughly first 100 vs last 100 are different
    # HMM doesn't guarantee 0 is low vol, so check segment coherence
    assert np.mean(preds[:100] == preds[0]) > 0.8
    assert np.mean(preds[100:] == preds[-1]) > 0.8
