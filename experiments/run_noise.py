from src.noise_models import create_noise_model
from src.quantum_reservoir import QuantumReservoir

def run_noise_experiment(X_train, y_train, X_test, y_test):
    noise_levels = [0.0, 0.005, 0.01, 0.02]

    results = []

    for p in noise_levels:
        print(f"Noise level: {p}")

        noise_model = create_noise_model(p)
        qrc = QuantumReservoir(noise_model=noise_model)

        X_train_q = [qrc.get_features(x) for x in X_train]
        X_test_q = [qrc.get_features(x) for x in X_test]

        from sklearn.linear_model import Ridge
        model = Ridge(alpha=0.01)
        model.fit(X_train_q, y_train)

        preds = model.predict(X_test_q)

        from src.evaluation import rmse
        error = rmse(y_test, preds)

        results.append((p, error))

    return results
