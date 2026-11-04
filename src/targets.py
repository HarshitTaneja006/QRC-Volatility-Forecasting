"""Volatility-target construction with explicit forecast timing."""

import numpy as np
import pandas as pd


_OHLC_COLUMNS = ("Open", "High", "Low", "Close")


def garman_klass_variance(ohlc: pd.DataFrame) -> pd.Series:
    """Return daily Garman-Klass variance from strictly positive OHLC prices."""
    missing = set(_OHLC_COLUMNS).difference(ohlc.columns)
    if missing:
        raise ValueError(f"OHLC data missing required columns: {sorted(missing)}")

    prices = ohlc.loc[:, _OHLC_COLUMNS].astype(float)
    if (prices <= 0).any().any():
        raise ValueError("OHLC prices must be strictly positive")

    high_low = np.log(prices["High"] / prices["Low"])
    close_open = np.log(prices["Close"] / prices["Open"])
    variance = 0.5 * high_low.pow(2) - (2 * np.log(2) - 1) * close_open.pow(2)
    return variance.rename("gk_variance")


def make_next_day_variance_target(variance: pd.Series) -> pd.Series:
    """Align each feature date with variance realized on its next trading date."""
    return variance.shift(-1).rename("next_day_gk_variance")
