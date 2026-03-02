import numpy as np
import matplotlib.pyplot as plt
from src.quantum_reservoir import QuantumReservoir
from src.evaluation import rmse

def run_scaling(X_train, y_train, X_test, y_test):
    results = []

    for n in [2,3,4,5,6,7,8]:
        print(f"Running {n} qubits...")

        qrc = QuantumReservoir(n_qubits=n)

        X_train_q = [qrc.get_features(x[:n]) for x in X_train]
        X_test_q = [qrc.get_features(x[:n]) for x in X_test]

        from sklearn.linear_model import Ridge
        model = Ridge(alpha=0.01)
        model.fit(X_train_q, y_train)

        preds = model.predict(X_test_q)
        error = rmse(y_test, preds)

        results.append((n, error))

    return results
