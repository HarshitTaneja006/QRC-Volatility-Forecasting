import numpy as np
import pandas as pd

from src.evaluation import expanding_window_forecast


class MeanModel:
    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.repeat(self.mean_, len(X))


def test_expanding_window_forecast_only_trains_on_past_rows():
    index = pd.bdate_range("2020-01-02", periods=5)
    X = pd.DataFrame({"lag_0": [1, 2, 3, 4, 5]}, index=index)
    y = pd.Series([10, 20, 30, 40, 50], index=index)

    predictions = expanding_window_forecast(
        X, y, estimator_factory=MeanModel, initial_train_size=2
    )

    assert predictions.index.equals(index[2:])
    assert predictions.tolist() == [15.0, 20.0, 25.0]
