"""
Deep Learning and Anomaly Detection models for Credit Card Fraud Detection.
Contains:
- AutoencoderScratch: Deep neural network autoencoder from scratch (NumPy)
- IsolationForestScratch: Isolation Forest anomaly detector from scratch (NumPy)
- AutoencoderIsolationForest: Proposed hybrid AE + IF model from the research paper
"""

from src.dl.autoencoder import AutoencoderScratch
from src.dl.isolation_forest import IsolationForestScratch
from src.dl.hybrid_ae_if import AutoencoderIsolationForest

__all__ = [
    "AutoencoderScratch",
    "IsolationForestScratch",
    "AutoencoderIsolationForest",
]
