import numpy as np
from sklearn.linear_model import Ridge
from .preprocessing import create_windows
from .metrics import rmse

def run_qrc(train, test, config, reservoir):

    X_train, y_train = create_windows(train, config["window_size"])
    X_test, y_test = create_windows(test, config["window_size"])

    X_train_q = [reservoir.get_features(x) for x in X_train]
    X_test_q = [reservoir.get_features(x) for x in X_test]

    model = Ridge(alpha=config["ridge_alpha"])
    model.fit(X_train_q, y_train)

    preds = model.predict(X_test_q)

    return y_test, preds
