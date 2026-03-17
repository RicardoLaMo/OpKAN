from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

class PCAReducer:
    """
    Standardizes and reduces feature dimensions via PCA.
    Maintains specified variance threshold.
    """
    def __init__(self, variance_threshold: float = 0.95):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=variance_threshold)
        self.is_fitted = False

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Standardize and fit PCA on the feature set."""
        X_scaled = self.scaler.fit_transform(X)
        X_pca = self.pca.fit_transform(X_scaled)
        self.is_fitted = True
        return X_pca

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Standardize and apply fitted PCA transformation."""
        if not self.is_fitted:
            raise ValueError("PCA model not fitted. Call fit_transform first.")
        X_scaled = self.scaler.transform(X)
        return self.pca.transform(X_scaled)

    def get_explained_variance(self) -> float:
        """Return the sum of explained variance of the components."""
        return np.sum(self.pca.explained_variance_ratio_)
