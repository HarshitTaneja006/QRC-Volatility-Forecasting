import numpy as np
import pandas as pd

from src.pipeline import prepare_growth_proxy_log_frame, prepare_spy_log_variance_frame


def test_prepare_spy_frame_targets_next_day_log_variance():
    index = pd.bdate_range("2020-01-02", periods=5)
    ohlc = pd.DataFrame(
        {
            "Open": [100, 101, 102, 103, 104],
            "High": [102, 104, 105, 107, 108],
            "Low": [99, 100, 101, 102, 103],
            "Close": [101, 103, 104, 106, 107],
        },
        index=index,
    )

    frame = prepare_spy_log_variance_frame(ohlc, lookback=2, variance_floor=1e-12)

    assert frame.index.equals(index[1:-1])
    assert np.isfinite(frame.to_numpy()).all()
    assert list(frame.columns) == ["lag_1", "lag_0", "target"]


def test_prepare_growth_proxy_uses_next_period_absolute_growth():
    index = pd.date_range("2020-01-01", periods=5, freq="MS")
    growth = pd.Series([-0.1, 0.2, -0.3, 0.4, -0.5], index=index)

    frame = prepare_growth_proxy_log_frame(growth, lookback=2, floor=1e-6)

    assert frame.index.equals(index[1:-1])
    assert frame.loc[index[1], "target"] == np.log(0.3)


def test_prepare_close_return_variance_frame_has_next_day_target():
    from src.pipeline import prepare_close_return_log_variance_frame

    ohlc = pd.DataFrame(
        {"Open": [10, 11, 12, 13, 14, 15], "High": [11, 12, 13, 14, 15, 16], "Low": [9, 10, 11, 12, 13, 14], "Close": [10, 11, 12, 13, 14, 15]},
        index=pd.date_range("2020-01-01", periods=6, freq="D"),
    )

    frame = prepare_close_return_log_variance_frame(ohlc, lookback=2, variance_floor=1e-12)

    assert list(frame.columns) == ["lag_1", "lag_0", "target"]
    assert frame.index.max() < ohlc.index.max()
