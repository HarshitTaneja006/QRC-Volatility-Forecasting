import numpy as np
import pandas as pd
import pytest

from src.targets import garman_klass_variance, make_next_day_variance_target


def test_garman_klass_variance_matches_closed_form():
    ohlc = pd.DataFrame(
        {
            "Open": [100.0],
            "High": [110.0],
            "Low": [95.0],
            "Close": [105.0],
        },
        index=pd.to_datetime(["2020-01-02"]),
    )

    result = garman_klass_variance(ohlc)
    expected = 0.5 * np.log(110.0 / 95.0) ** 2 - (
        2 * np.log(2) - 1
    ) * np.log(105.0 / 100.0) ** 2

    assert result.iloc[0] == pytest.approx(expected)


def test_next_day_target_is_shifted_without_future_feature_leakage():
    index = pd.bdate_range("2020-01-02", periods=4)
    variance = pd.Series([0.01, 0.02, 0.03, 0.04], index=index, name="gk_variance")

    target = make_next_day_variance_target(variance)

    assert target.iloc[0] == pytest.approx(0.02)
    assert target.iloc[2] == pytest.approx(0.04)
    assert np.isnan(target.iloc[-1])
