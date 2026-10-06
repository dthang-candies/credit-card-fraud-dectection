"""
Gaussian Naive Bayes Classifier from Scratch using NumPy.
Computes class priors, feature means, and variances with log-likelihood inference.
"""

from typing import Optional, Union, Dict
import numpy as np


class GaussianNaiveBayesScratch:
    """
    Gaussian Naive Bayes classifier from scratch.

    Attributes:
        var_smoothing: Portion of the largest variance added to all variances for calculation stability.
    """

    def __init__(self, var_smoothing: float = 1e-9):
        self.var_smoothing = var_smoothing
        self.classes: Optional[np.ndarray] = None
        self.priors: Dict[int, float] = {}
        self.means: Dict[int, np.ndarray] = {}
        self.vars: Dict[int, np.ndarray] = {}

    def fit(self, X: Union[np.ndarray, list], y: Union[np.ndarray, list]):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int).ravel()

        self.classes = np.unique(y)
        n_samples = len(y)
        global_var = np.var(X, axis=0)
        epsilon = self.var_smoothing * np.max(global_var)

        for c in self.classes:
            X_c = X[y == c]
            self.priors[c] = len(X_c) / n_samples
            self.means[c] = np.mean(X_c, axis=0)
            self.vars[c] = np.var(X_c, axis=0) + epsilon

        return self

    def _calculate_log_likelihood(self, X: np.ndarray, c: int) -> np.ndarray:
        """
        Compute log P(X | Y=c) for Gaussian features:
        -0.5 * sum( log(2 * pi * var) + ((x - mean)^2 / var) )
        """
        mean = self.means[c]
        var = self.vars[c]
        log_gaussian = -0.5 * np.sum(np.log(2.0 * np.pi * var) + ((X - mean) ** 2) / var, axis=1)
        return log_gaussian

    def predict_proba(self, X: Union[np.ndarray, list]) -> np.ndarray:
        """
        Predict posterior probabilities P(Y=1 | X).
        """
        X = np.asarray(X, dtype=float)
        log_posteriors = []

        for c in self.classes:
            log_prior = np.log(self.priors[c] + 1e-15)
            log_lik = self._calculate_log_likelihood(X, c)
            log_posteriors.append(log_prior + log_lik)

        # log_posteriors shape: (num_classes, num_samples) -> transpose to (num_samples, num_classes)
        log_posts = np.vstack(log_posteriors).T

        # Softmax with log-sum-exp trick
        max_log = np.max(log_posts, axis=1, keepdims=True)
        exp_post = np.exp(log_posts - max_log)
        probas = exp_post / np.sum(exp_post, axis=1, keepdims=True)

        # Return probability of class 1 (fraud)
        class_1_idx = np.where(self.classes == 1)[0][0]
        return probas[:, class_1_idx]

    def predict(self, X: Union[np.ndarray, list], threshold: float = 0.5) -> np.ndarray:
        probas = self.predict_proba(X)
        return (probas >= threshold).astype(int)

    def score(self, X, y) -> float:
        y_pred = self.predict(X)
        return float(np.mean(y_pred == np.asarray(y)))
