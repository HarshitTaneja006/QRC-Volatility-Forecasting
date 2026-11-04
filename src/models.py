"""Leakage-safe regression adapters for volatility experiments."""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .quantum_reservoir import QuantumReservoir


class _ScaledRidgeRegressor:
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.scaler = StandardScaler()
        self.readout = Ridge(alpha=alpha)

    def _features(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if X.ndim != 2 or len(X) != len(y):
            raise ValueError("X must be two-dimensional and aligned with y")
        features = self._features(self.scaler.fit_transform(X))
        self.readout.fit(features, y)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be two-dimensional")
        return self.readout.predict(self._features(self.scaler.transform(X)))


class QRCRegressor(_ScaledRidgeRegressor):
    """Ridge readout over fixed quantum-reservoir features."""

    def __init__(self, reservoir: QuantumReservoir, alpha: float = 1.0):
        super().__init__(alpha)
        self.reservoir = reservoir

    def _features(self, X: np.ndarray) -> np.ndarray:
        if X.shape[1] != self.reservoir.n_qubits:
            raise ValueError("QRC input width must equal reservoir.n_qubits")
        return self.reservoir.transform(X)


class CosineRidgeRegressor(_ScaledRidgeRegressor):
    """Classical control reproducing the old repository's cosine feature map."""

    def __init__(self, input_scale: float, alpha: float = 1.0):
        super().__init__(alpha)
        self.input_scale = float(input_scale)

    def _features(self, X: np.ndarray) -> np.ndarray:
        return np.cos(self.input_scale * X)
