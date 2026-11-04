"""Dataset assembly for causal volatility forecasting."""

import numpy as np
import pandas as pd

from .preprocessing import make_lagged_supervised
from .targets import garman_klass_variance, make_next_day_variance_target


def prepare_spy_log_variance_frame(
    ohlc: pd.DataFrame, *, lookback: int, variance_floor: float
) -> pd.DataFrame:
    """Create lagged log-variance features and a next-day Garman-Klass target."""
    if variance_floor <= 0:
        raise ValueError("variance_floor must be strictly positive")
    log_variance = np.log(garman_klass_variance(ohlc).clip(lower=variance_floor))
    next_day_log_variance = np.log(
        make_next_day_variance_target(garman_klass_variance(ohlc)).clip(
            lower=variance_floor
        )
    )
    return make_lagged_supervised(log_variance, next_day_log_variance, lookback)


def prepare_close_return_log_variance_frame(
    ohlc: pd.DataFrame, *, lookback: int, variance_floor: float
) -> pd.DataFrame:
    """Create a matched close-to-close conditional-variance robustness target."""
    if variance_floor <= 0:
        raise ValueError("variance_floor must be strictly positive")
    close = pd.Series(ohlc["Close"], dtype=float)
    close_variance = np.log(close / close.shift(1)).pow(2).clip(lower=variance_floor)
    log_variance = np.log(close_variance)
    return make_lagged_supervised(log_variance, log_variance.shift(-1), lookback)


def prepare_growth_proxy_log_frame(series, *, lookback: int, floor: float) -> pd.DataFrame:
    """Create robustness-test data for next-period absolute growth magnitude."""
    if floor <= 0:
        raise ValueError("floor must be strictly positive")
    log_proxy = np.log(pd.Series(series, dtype=float).abs().clip(lower=floor))
    return make_lagged_supervised(log_proxy, log_proxy.shift(-1), lookback)
