"""Leakage-safe tuning and uncertainty tools for the strengthened study."""

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from .metrics import mae, qlike, rmse


def rolling_origin_splits(
    n_samples: int, *, n_splits: int, validation_size: int, min_train_size: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return expanding chronological train/validation index pairs."""
    if n_splits < 1 or validation_size < 1 or min_train_size < 1:
        raise ValueError("split sizes must be positive")
    first_validation = n_samples - n_splits * validation_size
    if first_validation < min_train_size:
        raise ValueError("not enough observations for requested rolling-origin splits")
    return [
        (np.arange(0, validation_start), np.arange(validation_start, validation_start + validation_size))
        for validation_start in range(first_validation, n_samples, validation_size)
    ]


def select_by_inner_qlike(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    candidates: Sequence[dict],
    estimator_factory: Callable[[dict], object],
    n_splits: int,
    validation_size: int,
    min_train_size: int,
) -> tuple[dict, float]:
    """Choose parameters using only rolling inner validation folds."""
    splits = rolling_origin_splits(
        len(X), n_splits=n_splits, validation_size=validation_size, min_train_size=min_train_size
    )
    losses: list[float] = []
    for parameters in candidates:
        fold_losses = []
        for train, validation in splits:
            model = estimator_factory(parameters)
            model.fit(X.iloc[train], y.iloc[train])
            actual = np.exp(np.clip(y.iloc[validation].to_numpy(), -50, 50))
            predicted = np.exp(np.clip(np.asarray(model.predict(X.iloc[validation])), -50, 50))
            fold_losses.append(qlike(actual, predicted))
        losses.append(float(np.mean(fold_losses)))
    selected = int(np.argmin(losses))
    return dict(candidates[selected]), losses[selected]


def nested_expanding_forecast(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    candidate_selector: Callable[[pd.DataFrame, pd.Series], dict],
    estimator_factory: Callable[[dict], object],
    initial_train_size: int,
    refit_every: int,
) -> tuple[pd.Series, list[dict]]:
    """Forecast chronologically; choose parameters anew from each outer training window."""
    if not X.index.equals(y.index) or not X.index.is_monotonic_increasing:
        raise ValueError("X and y must have a shared monotonic chronological index")
    if not 0 < initial_train_size < len(X) or refit_every < 1:
        raise ValueError("invalid initial_train_size or refit_every")
    predictions: list[float] = []
    dates: list[pd.Timestamp] = []
    selected_parameters: list[dict] = []
    model = None
    for position in range(initial_train_size, len(X)):
        if model is None or (position - initial_train_size) % refit_every == 0:
            parameters = candidate_selector(X.iloc[:position], y.iloc[:position])
            model = estimator_factory(parameters)
            model.fit(X.iloc[:position], y.iloc[:position])
            selected_parameters.append(
                {
                    **parameters,
                    "forecast_start": str(X.index[position].date()),
                    "train_end": str(X.index[position - 1].date()),
                }
            )
        value = np.asarray(model.predict(X.iloc[[position]])).reshape(-1)
        if len(value) != 1 or not np.isfinite(value[0]):
            raise ValueError("estimator must produce one finite forecast")
        predictions.append(float(value[0]))
        dates.append(X.index[position])
    return pd.Series(predictions, index=pd.Index(dates), name="forecast"), selected_parameters


def moving_block_bootstrap_loss_difference(
    comparator_loss: np.ndarray,
    qrc_loss: np.ndarray,
    *,
    block_length: int = 10,
    n_bootstrap: int = 2_000,
    seed: int = 42,
) -> dict[str, float]:
    """Moving-block bootstrap for mean comparator-minus-QRC loss difference."""
    difference = np.asarray(comparator_loss, dtype=float) - np.asarray(qrc_loss, dtype=float)
    if difference.ndim != 1 or len(difference) < block_length:
        raise ValueError("loss arrays must be one-dimensional and at least one block long")
    if not np.isfinite(difference).all():
        raise ValueError("loss arrays must be finite")
    rng = np.random.default_rng(seed)
    n = len(difference)
    starts = np.arange(n - block_length + 1)
    draws = np.empty(n_bootstrap)
    for draw in range(n_bootstrap):
        blocks = rng.choice(starts, size=int(np.ceil(n / block_length)), replace=True)
        sample = np.concatenate([difference[start : start + block_length] for start in blocks])[:n]
        draws[draw] = sample.mean()
    p_value = 2.0 * min(float(np.mean(draws <= 0.0)), float(np.mean(draws >= 0.0)))
    return {
        "mean_difference": float(difference.mean()),
        "ci_lower": float(np.quantile(draws, 0.025)),
        "ci_upper": float(np.quantile(draws, 0.975)),
        "p_value_two_sided": min(1.0, p_value),
    }


def summarize_crisis_periods(
    forecasts: pd.DataFrame, periods: dict[str, tuple[str, str]]
) -> pd.DataFrame:
    """Compute model errors in named, inclusive date windows."""
    if "actual" not in forecasts:
        raise ValueError("forecasts must contain actual")
    records = []
    for period, (start, end) in periods.items():
        subset = forecasts.loc[start:end]
        if subset.empty:
            continue
        for model in subset.columns.drop("actual"):
            records.append(
                {
                    "period": period,
                    "model": model,
                    "n": len(subset),
                    "rmse": rmse(subset["actual"], subset[model]),
                    "mae": mae(subset["actual"], subset[model]),
                    "qlike": qlike(subset["actual"], subset[model]),
                }
            )
    return pd.DataFrame(records).set_index(["period", "model"]).sort_index()
