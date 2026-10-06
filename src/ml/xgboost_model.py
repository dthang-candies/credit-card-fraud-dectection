"""
XGBoost (Extreme Gradient Boosted Trees) Classifier from Scratch using NumPy.
Implements 2nd-order Taylor expansion gradient boosting with:
- First-order gradients g = p - y
- Second-order hessians h = p * (1 - p)
- Optimal leaf weights w = - sum(g) / (sum(h) + lambda)
- Tree split gain calculation with L2 regularization (lambda)
"""

from typing import Optional, Union, List
import numpy as np


class XGBoostNode:
    def __init__(
        self,
        feature_idx: Optional[int] = None,
        threshold: Optional[float] = None,
        left: Optional["XGBoostNode"] = None,
        right: Optional["XGBoostNode"] = None,
        *,
        weight: Optional[float] = None,
    ):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.weight = weight  # Leaf weight output

    @property
    def is_leaf(self) -> bool:
        return self.weight is not None


class XGBoostTree:
    """Single decision tree optimized for XGBoost gradient & hessian objective."""

    def __init__(
        self,
        max_depth: int = 4,
        reg_lambda: float = 1.0,
        gamma: float = 0.0,
        n_quantiles: int = 15,
    ):
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.n_quantiles = n_quantiles
        self.root: Optional[XGBoostNode] = None

    def _calc_leaf_weight(self, g: np.ndarray, h: np.ndarray) -> float:
        return - float(np.sum(g)) / (float(np.sum(h)) + self.reg_lambda)

    def _calc_gain(self, g_L, h_L, g_R, h_R, g_P, h_P) -> float:
        score_L = (g_L ** 2) / (h_L + self.reg_lambda)
        score_R = (g_R ** 2) / (h_R + self.reg_lambda)
        score_P = (g_P ** 2) / (h_P + self.reg_lambda)
        return 0.5 * (score_L + score_R - score_P) - self.gamma

    def fit(self, X: np.ndarray, g: np.ndarray, h: np.ndarray):
        self.root = self._build_tree(X, g, h, depth=0)
        return self

    def _build_tree(self, X: np.ndarray, g: np.ndarray, h: np.ndarray, depth: int) -> XGBoostNode:
        n_samples = len(g)
        leaf_weight = self._calc_leaf_weight(g, h)

        if depth >= self.max_depth or n_samples < 5:
            return XGBoostNode(weight=leaf_weight)

        best_gain = 0.0
        best_feat = None
        best_thresh = None

        g_total = np.sum(g)
        h_total = np.sum(h)
        n_features = X.shape[1]

        for feat_idx in range(n_features):
            vals = X[:, feat_idx]
            unique_vals = np.unique(vals)
            if len(unique_vals) <= 1:
                continue

            if len(unique_vals) > self.n_quantiles:
                percentiles = np.linspace(10, 90, self.n_quantiles)
                thresholds = np.percentile(vals, percentiles)
            else:
                thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0

            for thresh in thresholds:
                left_mask = vals <= thresh
                right_mask = ~left_mask

                if not np.any(left_mask) or not np.any(right_mask):
                    continue

                g_L = np.sum(g[left_mask])
                h_L = np.sum(h[left_mask])
                g_R = g_total - g_L
                h_R = h_total - h_L

                gain = self._calc_gain(g_L, h_L, g_R, h_R, g_total, h_total)

                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat_idx
                    best_thresh = thresh

        if best_gain <= 0.0 or best_feat is None:
            return XGBoostNode(weight=leaf_weight)

        left_mask = X[:, best_feat] <= best_thresh
        right_mask = ~left_mask

        left_child = self._build_tree(X[left_mask], g[left_mask], h[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], g[right_mask], h[right_mask], depth + 1)

        return XGBoostNode(
            feature_idx=best_feat,
            threshold=best_thresh,
            left=left_child,
            right=right_child,
        )

    def _predict_row(self, x: np.ndarray, node: XGBoostNode) -> float:
        if node.is_leaf:
            return node.weight
        if x[node.feature_idx] <= node.threshold:
            return self._predict_row(x, node.left)
        return self._predict_row(x, node.right)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._predict_row(x, self.root) for x in X])


class XGBoostClassifierScratch:
    """
    XGBoost Binary Classifier from scratch.

    Attributes:
        n_estimators: Number of boosting iterations (trees).
        learning_rate: Shrinkage step size (eta).
        max_depth: Maximum tree depth.
        reg_lambda: L2 regularization on leaf weights.
    """

    def __init__(
        self,
        n_estimators: int = 30,
        learning_rate: float = 0.1,
        max_depth: int = 4,
        reg_lambda: float = 1.0,
        gamma: float = 0.0,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.random_state = random_state

        self.trees: List[XGBoostTree] = []
        self.base_score: float = 0.0

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        clipped = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-clipped))

    def fit(self, X: Union[np.ndarray, list], y: Union[np.ndarray, list]):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        # Prior log-odds base score
        p_init = np.clip(np.mean(y), 1e-5, 1 - 1e-5)
        self.base_score = float(np.log(p_init / (1.0 - p_init)))

        raw_preds = np.full(len(y), self.base_score, dtype=float)
        self.trees = []

        for m in range(self.n_estimators):
            probas = self._sigmoid(raw_preds)

            # 1st and 2nd order gradients of logistic loss
            g = probas - y
            h = probas * (1.0 - probas)
            h = np.maximum(h, 1e-5)

            tree = XGBoostTree(
                max_depth=self.max_depth,
                reg_lambda=self.reg_lambda,
                gamma=self.gamma,
            )
            tree.fit(X, g, h)
            tree_preds = tree.predict(X)

            raw_preds += self.learning_rate * tree_preds
            self.trees.append(tree)

        return self

    def predict_proba(self, X: Union[np.ndarray, list]) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        raw_preds = np.full(len(X), self.base_score, dtype=float)

        for tree in self.trees:
            raw_preds += self.learning_rate * tree.predict(X)

        return self._sigmoid(raw_preds)

    def predict(self, X: Union[np.ndarray, list], threshold: float = 0.5) -> np.ndarray:
        probas = self.predict_proba(X)
        return (probas >= threshold).astype(int)

    def score(self, X, y) -> float:
        y_pred = self.predict(X)
        return float(np.mean(y_pred == np.asarray(y)))
