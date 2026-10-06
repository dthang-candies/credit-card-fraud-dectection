import numpy as np

class LogisticRegressionScratch:

    def __init__(
        self,
        learning_rate=0.01,
        n_iterations=1000,
        threshold=0.5
    ):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.threshold = threshold

        self.weights = None
        self.bias = None

        self.loss_history = []

    # ==========================================
    # Sigmoid
    # ==========================================

    def _sigmoid(self, z):

        # tránh overflow khi z quá lớn / quá nhỏ
        z = np.clip(z, -500, 500)

        return 1.0 / (1.0 + np.exp(-z))

    # ==========================================
    # Binary Cross Entropy
    # ==========================================

    def _compute_loss(self, y, y_prob):

        eps = 1e-15

        y_prob = np.clip(
            y_prob,
            eps,
            1 - eps
        )

        loss = -np.mean(
            y * np.log(y_prob)
            + (1 - y) * np.log(1 - y_prob)
        )

        return loss

    # ==========================================
    # Training
    # ==========================================

    def fit(self, X, y):

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        n_samples, n_features = X.shape

        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        self.loss_history = []

        for i in range(self.n_iterations):

            # ------------------------------
            # Forward propagation
            # ------------------------------

            linear = np.dot(X, self.weights) + self.bias

            y_prob = self._sigmoid(linear)

            # ------------------------------
            # Loss
            # ------------------------------

            loss = self._compute_loss(
                y,
                y_prob
            )

            self.loss_history.append(loss)

            # ------------------------------
            # Gradient
            # ------------------------------

            dw = (
                np.dot(
                    X.T,
                    (y_prob - y)
                )
                / n_samples
            )

            db = np.mean(
                y_prob - y
            )

            # ------------------------------
            # Gradient Descent
            # ------------------------------

            self.weights -= (
                self.learning_rate * dw
            )

            self.bias -= (
                self.learning_rate * db
            )

        return self

    # ==========================================
    # Predict probability
    # ==========================================

    def predict_proba(self, X):

        X = np.asarray(X, dtype=float)

        linear = (
            np.dot(X, self.weights)
            + self.bias
        )

        return self._sigmoid(linear)

    # ==========================================
    # Predict class
    # ==========================================

    def predict(self, X):

        probabilities = self.predict_proba(X)

        return (
            probabilities >= self.threshold
        ).astype(int)

    # ==========================================
    # Score
    # ==========================================

    def score(self, X, y):

        y_pred = self.predict(X)

        y = np.asarray(y)

        return np.mean(
            y_pred == y
        )