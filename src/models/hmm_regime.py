from hmmlearn.hmm import GaussianHMM
import numpy as np

class RegimeHMM:
    """
    Gaussian Hidden Markov Model for identifying market regimes.
    Regimes represent latent states (e.g., bull/bear, high/low vol).
    """
    def __init__(self, n_regimes: int = 3, covariance_type: str = 'diag'):
        self.model = GaussianHMM(
            n_components=n_regimes,
            covariance_type=covariance_type,
            n_iter=1000,
            random_state=42
        )
        self.is_fitted = False

    def fit(self, features: np.ndarray):
        """Fit HMM on pre-processed features."""
        # features should be (n_samples, n_features)
        self.model.fit(features)
        self.is_fitted = True
        return self

    def predict_regimes(self, features: np.ndarray) -> np.ndarray:
        """Decode most likely sequence of states (regimes)."""
        if not self.is_fitted:
            raise ValueError("HMM engine not fitted. Call fit first.")
        return self.model.predict(features)

    def get_regime_stats(self) -> dict:
        """Returns the transition matrix and means of each state."""
        if not self.is_fitted:
            raise ValueError("HMM engine not fitted.")
        return {
            'transition_matrix': self.model.transmat_,
            'means': self.model.means_,
            'covars': self.model.covars_
        }

def walk_forward_regime_inference(features: np.ndarray, train_window: int = 100) -> np.ndarray:
    """
    Simulates walk-forward inference by re-fitting the HMM as new data arrives.
    Helps in avoiding look-ahead bias.
    """
    n_samples = features.shape[0]
    regime_preds = np.zeros(n_samples)
    
    # Start inference after train_window
    for i in range(train_window, n_samples):
        window_features = features[i-train_window:i]
        hmm = RegimeHMM(n_regimes=2) # simplified to 2 regimes
        hmm.fit(window_features)
        
        # Predict the most recent regime
        # HMM is invariant to state labeling — sort states by mean realized_vol
        # (feature index 1) so regime 0 always maps to low-vol state across windows.
        # Falls back to feature 0 if the feature matrix has only one column.
        pred_seq = hmm.predict_regimes(window_features)
        n_feats = hmm.model.means_.shape[1]
        sort_idx = min(1, n_feats - 1)          # prefer realized_vol (col 1)
        means = hmm.model.means_[:, sort_idx]
        sorted_states = np.argsort(means)        # ascending: index 0 = lowest vol
        remap = {old: new for new, old in enumerate(sorted_states)}
        regime_preds[i] = remap[int(pred_seq[-1])]
        
    return regime_preds
