"""
Deep Learning Autoencoder from Scratch using NumPy.
Implements:
- Encoder: Input (d) -> Dense (h1) -> Bottleneck (z)
- Decoder: Bottleneck (z) -> Dense (h1) -> Output (d)
- Loss: Mean Squared Error (MSE) reconstruction loss
- Optimizer: Adam optimizer with mini-batch training
- Training on normal / legitimate transactions (unsupervised anomaly detection)
- Latent feature representation and reconstruction error scoring
"""

from typing import Optional, Union, List, Tuple
import numpy as np


class AutoencoderScratch:
    """
    Multilayer Perceptron Autoencoder from scratch using pure NumPy.

    Attributes:
        hidden_dim: Size of first hidden layer (e.g. 16 or 32).
        latent_dim: Size of bottleneck latent representation (e.g. 8).
        learning_rate: Adam optimizer learning rate.
        epochs: Number of training epochs.
        batch_size: Mini-batch size.
    """

    def __init__(
        self,
        hidden_dim: int = 16,
        latent_dim: int = 8,
        learning_rate: float = 0.001,
        epochs: int = 15,
        batch_size: int = 256,
        contamination: float = 0.002,
        random_state: int = 42,
    ):
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.contamination = contamination
        self.random_state = random_state

        # Weight matrices and biases
        self.W1: Optional[np.ndarray] = None  # (input_dim, hidden_dim)
        self.b1: Optional[np.ndarray] = None  # (hidden_dim,)
        self.W2: Optional[np.ndarray] = None  # (hidden_dim, latent_dim)
        self.b2: Optional[np.ndarray] = None  # (latent_dim,)
        self.W3: Optional[np.ndarray] = None  # (latent_dim, hidden_dim)
        self.b3: Optional[np.ndarray] = None  # (hidden_dim,)
        self.W4: Optional[np.ndarray] = None  # (hidden_dim, input_dim)
        self.b4: Optional[np.ndarray] = None  # (input_dim,)

        self.loss_history: List[float] = []
        self.threshold: float = 0.0

    def _relu(self, z: np.ndarray) -> np.ndarray:
        return np.maximum(0.01 * z, z)  # LeakyReLU

    def _relu_grad(self, z: np.ndarray) -> np.ndarray:
        grad = np.ones_like(z)
        grad[z < 0] = 0.01
        return grad

    def _init_weights(self, input_dim: int, rng: np.random.RandomState):
        # He / Xavier initialization
        self.W1 = rng.randn(input_dim, self.hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(self.hidden_dim)

        self.W2 = rng.randn(self.hidden_dim, self.latent_dim) * np.sqrt(2.0 / self.hidden_dim)
        self.b2 = np.zeros(self.latent_dim)

        self.W3 = rng.randn(self.latent_dim, self.hidden_dim) * np.sqrt(2.0 / self.latent_dim)
        self.b3 = np.zeros(self.hidden_dim)

        self.W4 = rng.randn(self.hidden_dim, input_dim) * np.sqrt(2.0 / self.hidden_dim)
        self.b4 = np.zeros(input_dim)

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Compress input X into latent representation Z."""
        z1 = np.dot(X, self.W1) + self.b1
        a1 = self._relu(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        return z2  # Bottleneck features

    def decode(self, Z: np.ndarray) -> np.ndarray:
        """Reconstruct X_hat from latent representation Z."""
        z3 = np.dot(Z, self.W3) + self.b3
        a3 = self._relu(z3)
        z4 = np.dot(a3, self.W4) + self.b4
        return z4  # Output reconstruction

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Layer 1 (Encoder Hidden)
        z1 = np.dot(X, self.W1) + self.b1
        a1 = self._relu(z1)
        # Layer 2 (Bottleneck)
        z2 = np.dot(a1, self.W2) + self.b2
        # Layer 3 (Decoder Hidden)
        z3 = np.dot(z2, self.W3) + self.b3
        a3 = self._relu(z3)
        # Layer 4 (Output Reconstruction)
        x_hat = np.dot(a3, self.W4) + self.b4
        return z1, a1, z2, z3, a3, x_hat

    def fit(self, X: Union[np.ndarray, list], y: Optional[Union[np.ndarray, list]] = None):
        """
        Train Autoencoder on normal transactions (Class=0) as outlined in the paper.
        """
        X_arr = np.asarray(X, dtype=float)
        rng = np.random.RandomState(self.random_state)

        # Train on legitimate (normal) data only if labels are provided
        if y is not None:
            y_arr = np.asarray(y).ravel()
            normal_mask = (y_arr == 0)
            X_train = X_arr[normal_mask]
        else:
            X_train = X_arr

        n_samples, input_dim = X_train.shape
        self._init_weights(input_dim, rng)

        # Adam optimizer moments
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        mW = {k: np.zeros_like(getattr(self, k)) for k in ["W1", "b1", "W2", "b2", "W3", "b3", "W4", "b4"]}
        vW = {k: np.zeros_like(getattr(self, k)) for k in ["W1", "b1", "W2", "b2", "W3", "b3", "W4", "b4"]}
        t = 0

        self.loss_history = []
        batch_size = min(self.batch_size, n_samples)

        for epoch in range(1, self.epochs + 1):
            perm = rng.permutation(n_samples)
            epoch_loss = 0.0

            for i in range(0, n_samples, batch_size):
                b_idx = perm[i : i + batch_size]
                X_b = X_train[b_idx]
                m = len(b_idx)

                # 1. Forward pass
                z1, a1, z2, z3, a3, x_hat = self.forward(X_b)
                diff = x_hat - X_b
                batch_loss = 0.5 * np.mean(diff ** 2)
                epoch_loss += batch_loss * m

                # 2. Backward pass
                # dL / dx_hat
                d_out = diff / m

                # Layer 4 gradients
                dW4 = np.dot(a3.T, d_out)
                db4 = np.sum(d_out, axis=0)

                # Backprop to Layer 3
                da3 = np.dot(d_out, self.W4.T)
                dz3 = da3 * self._relu_grad(z3)
                dW3 = np.dot(z2.T, dz3)
                db3 = np.sum(dz3, axis=0)

                # Backprop to Bottleneck
                dz2 = np.dot(dz3, self.W3.T)

                # Backprop to Layer 1
                da1 = np.dot(dz2, self.W2.T)
                dz1 = da1 * self._relu_grad(z1)
                dW1 = np.dot(X_b.T, dz1)
                db1 = np.sum(dz1, axis=0)
                dW2 = np.dot(a1.T, dz2)
                db2 = np.sum(dz2, axis=0)

                # 3. Adam parameter update
                t += 1
                grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3, "W4": dW4, "b4": db4}

                for param_name in ["W1", "b1", "W2", "b2", "W3", "b3", "W4", "b4"]:
                    g = grads[param_name]
                    mW[param_name] = beta1 * mW[param_name] + (1 - beta1) * g
                    vW[param_name] = beta2 * vW[param_name] + (1 - beta2) * (g ** 2)

                    m_hat = mW[param_name] / (1 - beta1 ** t)
                    v_hat = vW[param_name] / (1 - beta2 ** t)

                    current_param = getattr(self, param_name)
                    current_param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + eps)

            self.loss_history.append(epoch_loss / n_samples)

        # Calibrate anomaly threshold on training reconstruction errors
        train_errors = self.reconstruction_error(X_train)
        pct = 100.0 * (1.0 - self.contamination) if self.contamination < 0.5 else 99.0
        self.threshold = float(np.percentile(train_errors, pct))

        return self

    def reconstruct(self, X: Union[np.ndarray, list]) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        _, _, _, _, _, x_hat = self.forward(X)
        return x_hat

    def reconstruction_error(self, X: Union[np.ndarray, list]) -> np.ndarray:
        """Compute Mean Squared Error per sample: ||x - x_hat||^2."""
        X = np.asarray(X, dtype=float)
        x_hat = self.reconstruct(X)
        return np.mean((X - x_hat) ** 2, axis=1)

    def decision_function(self, X: Union[np.ndarray, list]) -> np.ndarray:
        """Returns reconstruction error as anomaly score."""
        return self.reconstruction_error(X)

    def predict_proba(self, X: Union[np.ndarray, list]) -> np.ndarray:
        """Min-max normalized reconstruction error into pseudo-probability."""
        errors = self.reconstruction_error(X)
        p_min, p_max = np.min(errors), np.max(errors)
        if p_max > p_min:
            return (errors - p_min) / (p_max - p_min)
        return np.zeros_like(errors)

    def predict(self, X: Union[np.ndarray, list], threshold: Optional[float] = None) -> np.ndarray:
        errors = self.reconstruction_error(X)
        thresh = threshold if threshold is not None else self.threshold
        return (errors >= thresh).astype(int)

    def score(self, X, y) -> float:
        y_pred = self.predict(X)
        return float(np.mean(y_pred == np.asarray(y)))


if __name__ == "__main__":
    import numpy as np
    print("Testing AutoencoderScratch...")
    X_dummy = np.random.randn(100, 10)
    ae = AutoencoderScratch(hidden_dim=8, latent_dim=4, epochs=5)
    ae.fit(X_dummy)
    latent = ae.encode(X_dummy)
    errors = ae.reconstruction_error(X_dummy)
    print("Latent shape:", latent.shape)
    print("Mean error:", np.mean(errors))
