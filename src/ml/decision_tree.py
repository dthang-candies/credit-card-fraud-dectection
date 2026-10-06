"""
Decision Tree Classifier from Scratch using NumPy.
Implements binary recursive splitting with Gini impurity and optional feature subsampling.
"""

from typing import Optional, Union, List
import numpy as np


class TreeNode:
    def __init__(
        self,
        feature_idx: Optional[int] = None,
        threshold: Optional[float] = None,
        left: Optional["TreeNode"] = None,
        right: Optional["TreeNode"] = None,
        *,
        value: Optional[float] = None,
        prob: float = 0.0,
    ):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # Class prediction
        self.prob = prob    # Probability of class 1

    @property
    def is_leaf(self) -> bool:
        return self.value is not None


class DecisionTreeClassifierScratch:
    """
    Binary Decision Tree Classifier from scratch.

    Attributes:
        max_depth: Maximum tree depth.
        min_samples_split: Minimum samples required to split an internal node.
        max_features: Number of features to consider when looking for best split (for Random Forest).
        n_quantiles: Max split candidates per feature to evaluate (for speed on continuous data).
    """

    def __init__(
        self,
        max_depth: int = 6,
        min_samples_split: int = 10,
        max_features: Optional[Union[int, float, str]] = None,
        n_quantiles: int = 20,
        random_state: int = 42,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.n_quantiles = n_quantiles
        self.random_state = random_state
        self.root: Optional[TreeNode] = None

    def fit(self, X: Union[np.ndarray, list], y: Union[np.ndarray, list]):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int).ravel()
        rng = np.random.RandomState(self.random_state)
        self.root = self._build_tree(X, y, depth=0, rng=rng)
        return self

    def _gini(self, y: np.ndarray) -> float:
        if len(y) == 0:
            return 0.0
        p1 = np.mean(y == 1)
        p0 = 1.0 - p1
        return 1.0 - (p0 ** 2 + p1 ** 2)

    def _best_split(self, X: np.ndarray, y: np.ndarray, rng: np.random.RandomState):
        best_gain = -1.0
        best_feat = None
        best_thresh = None

        n_samples, n_features = X.shape
        parent_gini = self._gini(y)

        # Feature subsampling (used by Random Forest)
        if self.max_features is None:
            feat_indices = np.arange(n_features)
        elif isinstance(self.max_features, str) and self.max_features == "sqrt":
            k = max(1, int(np.sqrt(n_features)))
            feat_indices = rng.choice(n_features, size=k, replace=False)
        elif isinstance(self.max_features, float):
            k = max(1, int(self.max_features * n_features))
            feat_indices = rng.choice(n_features, size=k, replace=False)
        else:
            k = min(n_features, int(self.max_features))
            feat_indices = rng.choice(n_features, size=k, replace=False)

        for feat_idx in feat_indices:
            vals = X[:, feat_idx]
            unique_vals = np.unique(vals)
            if len(unique_vals) <= 1:
                continue

            # Subsample split candidate thresholds via percentiles for high performance
            if len(unique_vals) > self.n_quantiles:
                percentiles = np.linspace(5, 95, self.n_quantiles)
                thresholds = np.percentile(vals, percentiles)
            else:
                thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0

            for thresh in thresholds:
                left_mask = vals <= thresh
                right_mask = ~left_mask

                n_left = np.sum(left_mask)
                n_right = len(y) - n_left

                if n_left == 0 or n_right == 0:
                    continue

                gain = (
                    parent_gini
                    - (n_left / n_samples) * self._gini(y[left_mask])
                    - (n_right / n_samples) * self._gini(y[right_mask])
                )

                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat_idx
                    best_thresh = thresh

        return best_feat, best_thresh, best_gain

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int, rng: np.random.RandomState) -> TreeNode:
        n_samples = len(y)
        p1 = float(np.mean(y == 1)) if n_samples > 0 else 0.0
        majority_class = int(p1 >= 0.5)

        # Stopping conditions
        if (
            depth >= self.max_depth
            or n_samples < self.min_samples_split
            or p1 == 0.0
            or p1 == 1.0
        ):
            return TreeNode(value=majority_class, prob=p1)

        feat_idx, thresh, gain = self._best_split(X, y, rng)

        if gain <= 1e-7 or feat_idx is None:
            return TreeNode(value=majority_class, prob=p1)

        left_mask = X[:, feat_idx] <= thresh
        right_mask = ~left_mask

        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1, rng)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1, rng)

        return TreeNode(
            feature_idx=feat_idx,
            threshold=thresh,
            left=left_child,
            right=right_child,
            value=None,
            prob=p1,
        )

    def _predict_row(self, x: np.ndarray, node: TreeNode) -> float:
        if node.is_leaf:
            return node.prob
        if x[node.feature_idx] <= node.threshold:
            return self._predict_row(x, node.left)
        return self._predict_row(x, node.right)

    def predict_proba(self, X: Union[np.ndarray, list]) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_row(x, self.root) for x in X])

    def predict(self, X: Union[np.ndarray, list], threshold: float = 0.5) -> np.ndarray:
        probas = self.predict_proba(X)
        return (probas >= threshold).astype(int)

    def score(self, X, y) -> float:
        y_pred = self.predict(X)
        return float(np.mean(y_pred == np.asarray(y)))
