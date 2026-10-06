"""
Hybrid Autoencoder + Isolation Forest Model for Credit Card Fraud Detection.
Implements the core proposed method in the paper:
'Phát hiện gian lận thẻ tín dụng sử dụng mô hình học sâu Autoencoder kết hợp thuật toán Isolation forest'
(Ngo Thuy Linh & Nguyen Duong Hung, 2024).

Architecture:
1. Autoencoder compresses high-dimensional transactions into latent bottleneck features (Z)
   and captures non-linear reconstruction discrepancies.
2. Isolation Forest isolates fraudulent transactions directly on the transformed feature space.
"""

from typing import Optional, Union, Tuple
import numpy as np
from src.dl.autoencoder import AutoencoderScratch
from src.dl.isolation_forest import IsolationForestScratch


class AutoencoderIsolationForest:
    """
    Proposed Hybrid Model combining Deep Autoencoder feature extraction
    with Isolation Forest anomaly detection.

    Attributes:
        ae_hidden_dim: Hidden dimension of Autoencoder.
        ae_latent_dim: Bottleneck latent dimension.
        ae_epochs: Number of Autoencoder epochs.
        if_n_estimators: Number of Isolation trees in forest (T).
        if_max_samples: Subsample size per tree (psi).
        contamination: Proportion of suspected fraud in data.
        include_recon_error: Whether to append the reconstruction error column to latent vector.
    """

    def __init__(
        self,
        ae_hidden_dim: int = 16,
        ae_latent_dim: int = 8,
        ae_epochs: int = 15,
        ae_lr: float = 0.001,
        if_n_estimators: int = 100,
        if_max_samples: Union[int, float] = 256,
        contamination: float = 0.002,
        include_recon_error: bool = True,
        random_state: int = 42,
    ):
        self.ae_hidden_dim = ae_hidden_dim
        self.ae_latent_dim = ae_latent_dim
        self.ae_epochs = ae_epochs
        self.ae_lr = ae_lr
        self.if_n_estimators = if_n_estimators
        self.if_max_samples = if_max_samples
        self.contamination = contamination
        self.include_recon_error = include_recon_error
        self.random_state = random_state

        self.autoencoder = AutoencoderScratch(
            hidden_dim=ae_hidden_dim,
            latent_dim=ae_latent_dim,
            epochs=ae_epochs,
            learning_rate=ae_lr,
            contamination=contamination,
            random_state=random_state,
        )

        self.isolation_forest = IsolationForestScratch(
            n_estimators=if_n_estimators,
            max_samples=if_max_samples,
            contamination=contamination,
            random_state=random_state,
        )

        self.threshold: float = 0.5

    def _extract_features(self, X: np.ndarray) -> np.ndarray:
        """
        Transform raw/scaled transactions into latent features + reconstruction error.
        """
        latent = self.autoencoder.encode(X)
        if self.include_recon_error:
            errors = self.autoencoder.reconstruction_error(X).reshape(-1, 1)
            # Scale error for numerical consistency with latent features
            scaled_errors = np.log1p(np.maximum(0.0, errors))
            return np.hstack([latent, scaled_errors])
        return latent

    def fit(self, X: Union[np.ndarray, list], y: Optional[Union[np.ndarray, list]] = None):
        """
        Fit the hybrid model:
        1. Fit Autoencoder on normal transactions (y == 0 if provided).
        2. Transform training data into latent representations.
        3. Fit Isolation Forest on the compressed representations.
        """
        X_arr = np.asarray(X, dtype=float)

        print("[AE+IF] Bước 1: Huấn luyện mạng học sâu Autoencoder để trích xuất đặc trưng...")
        self.autoencoder.fit(X_arr, y=y)

        print("[AE+IF] Bước 2: Nén dữ liệu đầu vào qua không gian tiềm ẩn (Bottleneck)...")
        Z_train = self._extract_features(X_arr)

        print("[AE+IF] Bước 3: Huấn luyện thuật toán Isolation Forest trên không gian tiềm ẩn...")
        self.isolation_forest.fit(Z_train)
        self.threshold = self.isolation_forest.threshold

        return self

    def decision_function(self, X: Union[np.ndarray, list]) -> np.ndarray:
        """
        Compute anomaly score s(z, n) via Isolation Forest on Autoencoder latent features.
        """
        X_arr = np.asarray(X, dtype=float)
        Z = self._extract_features(X_arr)
        return self.isolation_forest.decision_function(Z)

    def predict_proba(self, X: Union[np.ndarray, list]) -> np.ndarray:
        return self.decision_function(X)

    def predict(self, X: Union[np.ndarray, list], threshold: Optional[float] = None) -> np.ndarray:
        """
        Predict binary fraud classification: 1 = Gian lận, 0 = Bình thường.
        """
        scores = self.decision_function(X)
        thresh = threshold if threshold is not None else self.threshold
        return (scores >= thresh).astype(int)

    def score(self, X, y) -> float:
        y_pred = self.predict(X)
        return float(np.mean(y_pred == np.asarray(y)))
