"""Metrics and statistical-comparison tables for saved forecast panels."""

import pandas as pd

from .metrics import diebold_mariano, mae, qlike, rmse


def summarize_forecasts(forecasts: pd.DataFrame) -> pd.DataFrame:
    if "actual" not in forecasts:
        raise ValueError("forecasts must contain an actual column")
    actual = forecasts["actual"]
    rows = {}
    for name in forecasts.columns:
        if name == "actual":
            continue
        rows[name] = {
            "rmse": rmse(actual, forecasts[name]),
            "mae": mae(actual, forecasts[name]),
            "qlike": qlike(actual, forecasts[name]),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def summarize_magnitude_forecasts(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Summarize non-variance volatility proxies without QLIKE."""
    if "actual" not in forecasts:
        raise ValueError("forecasts must contain an actual column")
    actual = forecasts["actual"]
    rows = {}
    for name in forecasts.columns:
        if name == "actual":
            continue
        rows[name] = {"rmse": rmse(actual, forecasts[name]), "mae": mae(actual, forecasts[name])}
    return pd.DataFrame.from_dict(rows, orient="index")


def dm_against_qrc(forecasts: pd.DataFrame) -> pd.DataFrame:
    if "qrc" not in forecasts or "actual" not in forecasts:
        raise ValueError("forecasts must contain actual and qrc columns")
    rows = {}
    for name in forecasts.columns:
        if name in {"actual", "qrc"}:
            continue
        try:
            statistic, p_value = diebold_mariano(
                forecasts["actual"], forecasts[name], forecasts["qrc"]
            )
        except ValueError as error:
            if str(error) != "loss differential has zero long-run variance":
                raise
            statistic, p_value = float("nan"), float("nan")
        rows[name] = {"dm_statistic": statistic, "p_value": p_value}
    return pd.DataFrame.from_dict(rows, orient="index")
