"""
Isolation Forest Algorithm from Scratch using NumPy.
Strictly implements Algorithms 1, 2, 3, and 4 from the paper:
- Algorithm 1: Building Isolation Forest (iForest)
- Algorithm 2: Building Isolation Tree (iTree)
- Algorithm 3: Computing Anomaly Score s(x, n) = 2^(-E(h(x)) / c(n))
- Algorithm 4: Computing Path Length PathLength(x, Tree)
"""

from typing import Optional, Union, List
import numpy as np


def c_factor(n: int) -> float:
    """
    Average path length of unsuccessful search in a Binary Search Tree (BST)
    given n data points:
    c(n) = 2 * (ln(n - 1) + 0.5772156649) - 2 * (n - 1) / n
    Euler-Mascheroni constant = 0.5772156649
    """
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    euler_mascheroni = 0.5772156649
    return 2.0 * (np.log(n - 1) + euler_mascheroni) - (2.0 * (n - 1) / n)


class IsolationTreeNode:
    """Node in an Isolation Tree (iTree)."""

    def __init__(
        self,
        left: Optional["IsolationTreeNode"] = None,
        right: Optional["IsolationTreeNode"] = None,
        split_att: Optional[int] = None,
        split_val: Optional[float] = None,
        size: int = 0,
        node_type: str = "internal",  # 'internal' or 'external'
    ):
        self.left = left
        self.right = right
        self.split_att = split_att
        self.split_val = split_val
        self.size = size
        self.node_type = node_type

    @property
    def is_external(self) -> bool:
        return self.node_type == "external"


class IsolationForestScratch:
    """
    Isolation Forest Anomaly Detector from scratch.

    Attributes:
        n_estimators: Number of isolation trees (T).
        max_samples: Subsampling size (psi). Standard default is 256.
        contamination: Estimated proportion of outliers / fraud in the dataset (used for thresholding).
        random_state: Random state for reproducibility.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_samples: Union[int, float] = 256,
        contamination: float = 0.002,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.random_state = random_state

        self.trees: List[IsolationTreeNode] = []
        self.subsample_size: int = 256
        self.threshold: float = 0.5
        self.c_subsample: float = 1.0

    def _build_itree(
        self,
        X: np.ndarray,
        current_height: int,
        height_limit: int,
        rng: np.random.RandomState
    ) -> IsolationTreeNode:
        """
        Algorithm 2: Xây dựng cây cô lập iTree.
        """
        n_samples, n_features = X.shape

        # Step 1: If current_height >= height_limit or |X| <= 1: return exNode(size=|X|)
        if current_height >= height_limit or n_samples <= 1:
            return IsolationTreeNode(size=n_samples, node_type="external")

        # Step 2: Pick random attribute q in Q
        q = rng.randint(0, n_features)
        col_vals = X[:, q]
        min_val = np.min(col_vals)
        max_val = np.max(col_vals)

        if min_val >= max_val:
            return IsolationTreeNode(size=n_samples, node_type="external")

        # Pick random split value p in [min, max]
        p = rng.uniform(min_val, max_val)

        left_mask = col_vals < p
        right_mask = ~left_mask

        left_child = self._build_itree(X[left_mask], current_height + 1, height_limit, rng)
        right_child = self._build_itree(X[right_mask], current_height + 1, height_limit, rng)

        return IsolationTreeNode(
            left=left_child,
            right=right_child,
            split_att=q,
            split_val=p,
            size=n_samples,
            node_type="internal",
        )

    def fit(self, X: Union[np.ndarray, list], y=None):
        """
        Algorithm 1: Xây dựng Isolation Forest.
        """
        X = np.asarray(X, dtype=float)
        n_samples = len(X)

        if isinstance(self.max_samples, float):
            self.subsample_size = int(self.max_samples * n_samples)
        else:
            self.subsample_size = min(int(self.max_samples), n_samples)

        # Height limit L = ceiling(log2(subsample_size))
        height_limit = int(np.ceil(np.log2(max(2, self.subsample_size))))
        self.c_subsample = c_factor(self.subsample_size)

        rng = np.random.RandomState(self.random_state)
        self.trees = []

        for _ in range(self.n_estimators):
            tree_rng = np.random.RandomState(rng.randint(0, 1_000_000))
            sub_indices = tree_rng.choice(n_samples, size=self.subsample_size, replace=False)
            X_sub = X[sub_indices]
            tree = self._build_itree(X_sub, current_height=0, height_limit=height_limit, rng=tree_rng)
            self.trees.append(tree)

        # Calibrate decision threshold based on training anomaly scores and contamination
        train_scores = self.decision_function(X)
        if self.contamination is not None and 0.0 < self.contamination < 0.5:
            # Score near 1 means anomaly. Outliers are the top (1 - contamination) percentile.
            self.threshold = float(np.percentile(train_scores, 100.0 * (1.0 - self.contamination)))
        else:
            self.threshold = 0.5

        return self

    def _path_length(self, x: np.ndarray, node: IsolationTreeNode, current_edge: int = 0) -> float:
        """
        Algorithm 4: Tính chiều cao của cây PathLength.
        """
        if node.is_external:
            return current_edge + c_factor(node.size)

        if x[node.split_att] < node.split_val:
            return self._path_length(x, node.left, current_edge + 1)
        else:
            return self._path_length(x, node.right, current_edge + 1)

    def decision_function(self, X: Union[np.ndarray, list]) -> np.ndarray:
        """
        Algorithm 3: Tính giá trị điểm số bất thường s(x, n) = 2^(-E(h(x)) / c(n)).
        Higher score in [0, 1] indicates higher likelihood of anomaly (fraud).
        """
        X = np.asarray(X, dtype=float)
        n_samples = len(X)
        all_paths = np.zeros((self.n_estimators, n_samples), dtype=float)

        for t_idx, tree in enumerate(self.trees):
            for i in range(n_samples):
                all_paths[t_idx, i] = self._path_length(X[i], tree, 0)

        # Average path length E(h(x))
        avg_paths = np.mean(all_paths, axis=0)

        # Anomaly score s(x, n)
        if self.c_subsample <= 0:
            return np.full(n_samples, 0.5)

        scores = 2.0 ** (- (avg_paths / self.c_subsample))
        return scores

    def predict_proba(self, X: Union[np.ndarray, list]) -> np.ndarray:
        """Return anomaly score as fraud probability."""
        return self.decision_function(X)

    def predict(self, X: Union[np.ndarray, list], threshold: Optional[float] = None) -> np.ndarray:
        """
        Predict binary labels: 1 = Fraud / Anomaly, 0 = Normal / Legitimate.
        """
        scores = self.decision_function(X)
        thresh = threshold if threshold is not None else self.threshold
        return (scores >= thresh).astype(int)

    def score(self, X, y) -> float:
        y_pred = self.predict(X)
        return float(np.mean(y_pred == np.asarray(y)))
