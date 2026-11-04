"""Model evaluation for log-variance forecasting."""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .evaluation import expanding_window_forecast
from .models import CosineRidgeRegressor, QRCRegressor
from .quantum_reservoir import QuantumReservoir
from .robustness import nested_expanding_forecast, select_by_inner_qlike


def _variance_from_log(log_forecast):
    return np.exp(np.clip(np.asarray(log_forecast, dtype=float), -50, 50))


def evaluate_log_variance_models(
    frame: pd.DataFrame,
    *,
    initial_train_size: int,
    refit_every: int,
    ridge_alpha: float,
    input_scale: float,
    qrc_layers: int,
) -> pd.DataFrame:
    """Evaluate controls and QRC with a common expanding-window forecast index."""
    if "target" not in frame:
        raise ValueError("frame must contain a target column")
    X = frame.drop(columns="target")
    y = frame["target"]
    if X.shape[1] < 2:
        raise ValueError("QRC requires at least two lag features")

    ridge = expanding_window_forecast(
        X,
        y,
        estimator_factory=lambda: make_pipeline(StandardScaler(), Ridge(alpha=ridge_alpha)),
        initial_train_size=initial_train_size,
        refit_every=refit_every,
    )
    cosine = expanding_window_forecast(
        X,
        y,
        estimator_factory=lambda: CosineRidgeRegressor(
            input_scale=input_scale, alpha=ridge_alpha
        ),
        initial_train_size=initial_train_size,
        refit_every=refit_every,
    )
    qrc = expanding_window_forecast(
        X,
        y,
        estimator_factory=lambda: QRCRegressor(
            QuantumReservoir(
                X.shape[1],
                input_scale=input_scale,
                layers=qrc_layers,
                include_pair_observables=True,
            ),
            alpha=ridge_alpha,
        ),
        initial_train_size=initial_train_size,
        refit_every=refit_every,
    )
    forecast_index = ridge.index
    naive = X.loc[forecast_index, "lag_0"].rename("naive")
    return pd.DataFrame(
        {
            "actual": _variance_from_log(y.loc[forecast_index]),
            "naive": _variance_from_log(naive),
            "ridge": _variance_from_log(ridge),
            "cosine": _variance_from_log(cosine),
            "qrc": _variance_from_log(qrc),
        },
        index=forecast_index,
    )


def evaluate_nested_log_variance_models(
    frame: pd.DataFrame,
    *,
    initial_train_size: int,
    refit_every: int,
    inner_splits: int,
    inner_validation_size: int,
    ridge_candidates: list[dict],
    cosine_candidates: list[dict],
    qrc_candidates: list[dict],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Nested rolling-origin selection for all learned models on one common task."""
    if "target" not in frame:
        raise ValueError("frame must contain target")
    X, y = frame.drop(columns="target"), frame["target"]
    if X.shape[1] < 2:
        raise ValueError("QRC requires at least two lag features")
    minimum_inner_training = max(2, inner_validation_size)

    factories = {
        "ridge": lambda parameters: make_pipeline(
            StandardScaler(), Ridge(alpha=parameters["alpha"])
        ),
        "cosine": lambda parameters: CosineRidgeRegressor(
            input_scale=parameters["input_scale"], alpha=parameters["alpha"]
        ),
        "qrc": lambda parameters: QRCRegressor(
            QuantumReservoir(
                X.shape[1],
                input_scale=parameters["input_scale"],
                layers=parameters["layers"],
                entangle=parameters.get("entangle", True),
                include_pair_observables=parameters.get("include_pair_observables", True),
                depolarizing_probability=parameters.get("depolarizing_probability", 0.0),
            ),
            alpha=parameters["alpha"],
        ),
    }
    candidates_by_model = {
        "ridge": ridge_candidates,
        "cosine": cosine_candidates,
        "qrc": qrc_candidates,
    }
    predictions: dict[str, pd.Series] = {}
    selections: dict[str, pd.DataFrame] = {}
    for name, candidates in candidates_by_model.items():
        factory = factories[name]

        def selector(train_x, train_y, *, _candidates=candidates, _factory=factory):
            return select_by_inner_qlike(
                train_x,
                train_y,
                candidates=_candidates,
                estimator_factory=_factory,
                n_splits=inner_splits,
                validation_size=inner_validation_size,
                min_train_size=minimum_inner_training,
            )[0]

        predictions[name], choice_log = nested_expanding_forecast(
            X,
            y,
            candidate_selector=selector,
            estimator_factory=factory,
            initial_train_size=initial_train_size,
            refit_every=refit_every,
        )
        selections[name] = pd.DataFrame(choice_log)
    forecast_index = predictions["ridge"].index
    naive = X.loc[forecast_index, "lag_0"].rename("naive")
    forecasts = pd.DataFrame(
        {
            "actual": _variance_from_log(y.loc[forecast_index]),
            "naive": _variance_from_log(naive),
            **{name: _variance_from_log(prediction) for name, prediction in predictions.items()},
        },
        index=forecast_index,
    )
    return forecasts, selections
