import numpy as np
from src.quantum_reservoir import QuantumReservoir
from src.evaluation import rmse
from sklearn.linear_model import Ridge


def run_qubit_scaling(X_train, y_train, X_test, y_test, qubit_range=None):
    """Run QRC at different qubit counts and return RMSE results."""
    if qubit_range is None:
        qubit_range = [2, 3, 4, 5, 6, 7, 8]

    results = []

    for n in qubit_range:
        qrc = QuantumReservoir(n_qubits=n)

        X_train_q = [qrc.get_features(x[:n]) for x in X_train]
        X_test_q = [qrc.get_features(x[:n]) for x in X_test]

        model = Ridge(alpha=0.01)
        model.fit(X_train_q, y_train)

        preds = model.predict(X_test_q)
        error = rmse(y_test, preds)
        results.append((n, error))

    return results
