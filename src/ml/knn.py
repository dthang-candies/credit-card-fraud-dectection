"""
K-Nearest Neighbors (KNN) Classifier from Scratch using NumPy.
Uses vectorized Euclidean distance and batch-wise neighbor search.
"""

from typing import Optional, Union
import numpy as np


class KNNScratch:
    """
    K-Nearest Neighbors Classifier from scratch.

    Attributes:
        n_neighbors: Number of nearest neighbors (k).
        max_train_samples: If training set is huge (e.g. >200k), optionally sample
                           representative points to maintain fast distance querying.
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        max_train_samples: Optional[int] = 10000,
        random_state: int = 42,
    ):
        self.n_neighbors = n_neighbors
        self.max_train_samples = max_train_samples
        self.random_state = random_state

        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None

    def fit(self, X: Union[np.ndarray, list], y: Union[np.ndarray, list]):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int).ravel()

        if self.max_train_samples and len(X) > self.max_train_samples:
            rng = np.random.RandomState(self.random_state)
            # Stratified subsample to preserve rare fraud class in neighbor memory
            pos_idx = np.where(y == 1)[0]
            neg_idx = np.where(y == 0)[0]

            n_pos = len(pos_idx)
            n_neg = self.max_train_samples - n_pos
            n_neg = max(100, min(n_neg, len(neg_idx)))

            sampled_neg = rng.choice(neg_idx, size=n_neg, replace=False)
            all_idx = np.concatenate([pos_idx, sampled_neg])
            rng.shuffle(all_idx)

            self.X_train = X[all_idx]
            self.y_train = y[all_idx]
        else:
            self.X_train = X
            self.y_train = y

        return self

    def _predict_batch_probas(self, X_batch: np.ndarray) -> np.ndarray:
        """
        Compute probabilities for a batch of query points using vectorized Euclidean distance.
        ||A - B||^2 = ||A||^2 - 2 A B^T + ||B||^2
        """
        # A: (B, D), B: (N, D)
        a2 = np.sum(X_batch ** 2, axis=1, keepdims=True)
        b2 = np.sum(self.X_train ** 2, axis=1, keepdims=True).T
        ab = np.dot(X_batch, self.X_train.T)
        dists = np.maximum(0.0, a2 - 2 * ab + b2)

        # Find top k smallest distances indices
        k_indices = np.argpartition(dists, self.n_neighbors, axis=1)[:, : self.n_neighbors]
        k_labels = self.y_train[k_indices]

        # Fraud probability is the proportion of positive labels among k neighbors
        probas = np.mean(k_labels == 1, axis=1)
        return probas

    def predict_proba(self, X: Union[np.ndarray, list], batch_size: int = 1000) -> np.ndarray:
        """
        Compute predicted probabilities for class 1 in chunks to avoid memory spikes.
        """
        X = np.asarray(X, dtype=float)
        n_samples = len(X)
        probas = np.zeros(n_samples, dtype=float)

        for i in range(0, n_samples, batch_size):
            end = min(i + batch_size, n_samples)
            probas[i:end] = self._predict_batch_probas(X[i:end])

        return probas

    def predict(
        self,
        X: Union[np.ndarray, list],
        threshold: float = 0.5,
        batch_size: int = 1000
    ) -> np.ndarray:
        probas = self.predict_proba(X, batch_size=batch_size)
        return (probas >= threshold).astype(int)

    def score(self, X, y) -> float:
        y_pred = self.predict(X)
        return float(np.mean(y_pred == np.asarray(y)))
