"""
Support Vector Machine (Linear SVM) from Scratch using NumPy.
Implements Soft-margin Hinge Loss with L2 Regularization and Subgradient Descent.
"""

from typing import Optional, Union
import numpy as np


class LinearSVMScratch:
    """
    Linear Support Vector Machine (Binary Classification) from scratch.

    Attributes:
        learning_rate: Initial step size for subgradient descent.
        lambda_param: Regularization parameter (higher = stronger regularization).
        n_iterations: Number of epochs over the data.
        batch_size: Mini-batch size for stochastic subgradient descent.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        lambda_param: float = 0.01,
        n_iterations: int = 500,
        batch_size: int = 256,
        random_state: int = 42,
    ):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.random_state = random_state

        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0
        self.loss_history: list = []

    def fit(self, X: Union[np.ndarray, list], y: Union[np.ndarray, list]):
        """
        Fit the Linear SVM model using mini-batch subgradient descent.
        Converts labels {0, 1} to {-1, +1} for hinge loss.
        """
        rng = np.random.RandomState(self.random_state)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        # Convert {0, 1} to {-1, +1}
        y_svm = np.where(y <= 0, -1.0, 1.0)
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []

        batch_size = min(self.batch_size, n_samples)

        for epoch in range(1, self.n_iterations + 1):
            lr = self.learning_rate / (1.0 + 0.001 * epoch)
            indices = rng.permutation(n_samples)

            for start_idx in range(0, n_samples, batch_size):
                batch_idx = indices[start_idx : start_idx + batch_size]
                X_batch = X[batch_idx]
                y_batch = y_svm[batch_idx]
                k = len(batch_idx)

                # Functional margin: y * (w^T x + b)
                margins = y_batch * (np.dot(X_batch, self.weights) + self.bias)
                misclassified_mask = margins < 1.0

                # Gradients
                dw = self.lambda_param * self.weights
                db = 0.0

                if np.any(misclassified_mask):
                    dw -= np.dot(y_batch[misclassified_mask], X_batch[misclassified_mask]) / k
                    db -= np.sum(y_batch[misclassified_mask]) / k

                self.weights -= lr * dw
                self.bias -= lr * db

            if epoch % 50 == 0 or epoch == self.n_iterations:
                margins_all = y_svm * (np.dot(X, self.weights) + self.bias)
                hinge_loss = np.mean(np.maximum(0, 1.0 - margins_all))
                reg_loss = 0.5 * self.lambda_param * np.sum(self.weights ** 2)
                self.loss_history.append(hinge_loss + reg_loss)

        return self

    def decision_function(self, X: Union[np.ndarray, list]) -> np.ndarray:
        """
        Calculate signed distance to the hyperplane: w^T x + b.
        """
        X = np.asarray(X, dtype=float)
        return np.dot(X, self.weights) + self.bias

    def predict_proba(self, X: Union[np.ndarray, list]) -> np.ndarray:
        """
        Estimate fraud probability using Platt-scaling approximation (sigmoid of margins).
        """
        decisions = self.decision_function(X)
        clipped = np.clip(decisions, -500, 500)
        return 1.0 / (1.0 + np.exp(-clipped))

    def predict(self, X: Union[np.ndarray, list], threshold: float = 0.0) -> np.ndarray:
        """
        Predict binary labels (0 or 1).
        """
        decisions = self.decision_function(X)
        return (decisions >= threshold).astype(int)

    def score(self, X, y) -> float:
        y_pred = self.predict(X)
        return float(np.mean(y_pred == np.asarray(y)))
