"""Chronological forecasting evaluation utilities."""

from collections.abc import Callable

import numpy as np
import pandas as pd


def expanding_window_forecast(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    estimator_factory: Callable[[], object],
    initial_train_size: int,
    refit_every: int = 1,
) -> pd.Series:
    """Produce one-step forecasts using only rows strictly before each forecast row."""
    if not X.index.equals(y.index):
        raise ValueError("X and y must share the same chronological index")
    if not X.index.is_monotonic_increasing:
        raise ValueError("X index must be monotonic increasing")
    if initial_train_size < 1 or initial_train_size >= len(X):
        raise ValueError("initial_train_size must leave at least one forecast row")
    if refit_every < 1:
        raise ValueError("refit_every must be at least one")

    predictions = []
    forecast_index = []
    model = None
    for position in range(initial_train_size, len(X)):
        if model is None or (position - initial_train_size) % refit_every == 0:
            model = estimator_factory()
            model.fit(X.iloc[:position], y.iloc[:position])
        prediction = np.asarray(model.predict(X.iloc[[position]])).reshape(-1)
        if len(prediction) != 1 or not np.isfinite(prediction[0]):
            raise ValueError("estimator must return one finite prediction per row")
        predictions.append(float(prediction[0]))
        forecast_index.append(X.index[position])

    return pd.Series(predictions, index=pd.Index(forecast_index), name="forecast")
