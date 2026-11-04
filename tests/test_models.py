import numpy as np

from src.models import CosineRidgeRegressor, QRCRegressor
from src.quantum_reservoir import QuantumReservoir


def test_qrc_regressor_returns_one_prediction_per_row():
    X = np.array(
        [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5], [0.4, 0.5, 0.6]]
    )
    y = np.array([-4.0, -3.8, -3.5, -3.2])
    model = QRCRegressor(
        QuantumReservoir(3, input_scale=0.8, layers=1), alpha=0.1
    )

    model.fit(X[:3], y[:3])
    predictions = model.predict(X[3:])

    assert predictions.shape == (1,)
    assert np.isfinite(predictions).all()


def test_cosine_baseline_matches_training_and_prediction_interface():
    X = np.array([[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]])
    y = np.array([-4.0, -3.8, -3.6])
    model = CosineRidgeRegressor(input_scale=0.5, alpha=0.1)

    model.fit(X[:2], y[:2])

    assert model.predict(X[2:]).shape == (1,)
