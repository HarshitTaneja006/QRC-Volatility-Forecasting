import numpy as np
import pandas as pd

from src.robustness import nested_expanding_forecast


class MeanEstimator:
    def __init__(self, offset):
        self.offset = offset

    def fit(self, X, y):
        self.value = float(np.mean(y)) + self.offset
        return self

    def predict(self, X):
        return np.full(len(X), self.value)


def test_nested_forecast_selects_from_inner_training_only():
    index = pd.date_range("2020-01-01", periods=22, freq="D")
    X = pd.DataFrame({"x": np.arange(22, dtype=float)}, index=index)
    y = pd.Series(np.arange(22, dtype=float), index=index)
    seen_train_ends = []

    def selector(train_x, train_y):
        seen_train_ends.append(train_x.index[-1])
        return {"offset": 0.0}

    prediction, chosen = nested_expanding_forecast(
        X,
        y,
        candidate_selector=selector,
        estimator_factory=lambda parameters: MeanEstimator(parameters["offset"]),
        initial_train_size=10,
        refit_every=4,
    )

    assert prediction.index[0] == index[10]
    assert seen_train_ends == [index[9], index[13], index[17]]
    assert chosen[0]["forecast_start"] == str(index[10].date())
