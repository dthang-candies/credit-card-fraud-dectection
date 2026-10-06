"""
Random Forest Classifier from Scratch using NumPy.
Ensemble of DecisionTreeClassifierScratch using Bootstrap Aggregation (Bagging)
and Random Feature Subsampling.
"""

from typing import Optional, Union, List
import numpy as np
from src.ml.decision_tree import DecisionTreeClassifierScratch


class RandomForestClassifierScratch:
    """
    Random Forest Classifier from scratch.

    Attributes:
        n_estimators: Number of decision trees.
        max_depth: Maximum depth of each tree.
        min_samples_split: Minimum samples to split.
        max_features: Feature subsampling per split ('sqrt', float, or int).
        bootstrap: Whether to sample with replacement.
        max_samples: Number of samples per tree bootstrap.
    """

    def __init__(
        self,
        n_estimators: int = 25,
        max_depth: int = 8,
        min_samples_split: int = 10,
        max_features: Union[str, float, int] = "sqrt",
        bootstrap: bool = True,
        max_samples: Optional[int] = None,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.max_samples = max_samples
        self.random_state = random_state

        self.trees: List[DecisionTreeClassifierScratch] = []

    def fit(self, X: Union[np.ndarray, list], y: Union[np.ndarray, list]):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int).ravel()
        n_samples = len(X)
        sample_size = self.max_samples if self.max_samples else n_samples

        rng = np.random.RandomState(self.random_state)
        self.trees = []

        for i in range(self.n_estimators):
            tree_seed = rng.randint(0, 1_000_000)
            tree_rng = np.random.RandomState(tree_seed)

            # Bootstrap sampling
            if self.bootstrap:
                boot_idx = tree_rng.choice(n_samples, size=sample_size, replace=True)
            else:
                boot_idx = tree_rng.choice(n_samples, size=min(sample_size, n_samples), replace=False)

            tree = DecisionTreeClassifierScratch(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                random_state=tree_seed,
            )
            tree.fit(X[boot_idx], y[boot_idx])
            self.trees.append(tree)

        return self

    def predict_proba(self, X: Union[np.ndarray, list]) -> np.ndarray:
        """
        Ensemble probability is the average probability output across all trees.
        """
        X = np.asarray(X, dtype=float)
        all_tree_probas = np.array([tree.predict_proba(X) for tree in self.trees])
        return np.mean(all_tree_probas, axis=0)

    def predict(self, X: Union[np.ndarray, list], threshold: float = 0.5) -> np.ndarray:
        probas = self.predict_proba(X)
        return (probas >= threshold).astype(int)

    def score(self, X, y) -> float:
        y_pred = self.predict(X)
        return float(np.mean(y_pred == np.asarray(y)))
