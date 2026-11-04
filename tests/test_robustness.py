import numpy as np
import pandas as pd

from src.robustness import (
    moving_block_bootstrap_loss_difference,
    rolling_origin_splits,
    summarize_crisis_periods,
)


def test_rolling_origin_splits_never_use_validation_observations_for_training():
    splits = rolling_origin_splits(40, n_splits=3, validation_size=5, min_train_size=12)

    assert len(splits) == 3
    for train, validation in splits:
        assert len(train) >= 12
        assert train[-1] < validation[0]
        assert len(validation) == 5


def test_moving_block_bootstrap_reports_positive_difference_when_qrc_is_better():
    result = moving_block_bootstrap_loss_difference(
        np.full(40, 1.0), np.full(40, 0.5), block_length=5, n_bootstrap=200, seed=7
    )

    assert result["mean_difference"] == 0.5
    assert result["ci_lower"] == 0.5
    assert result["ci_upper"] == 0.5
    assert result["p_value_two_sided"] == 0.0


def test_crisis_summary_uses_requested_date_windows():
    index = pd.date_range("2020-01-01", periods=4, freq="D")
    forecasts = pd.DataFrame(
        {"actual": [1.0, 1.0, 4.0, 4.0], "ridge": [1.0, 2.0, 3.0, 5.0], "qrc": [1.0, 1.0, 4.0, 4.0]},
        index=index,
    )
    periods = {"first": ("2020-01-01", "2020-01-02"), "second": ("2020-01-03", "2020-01-04")}

    summary = summarize_crisis_periods(forecasts, periods)

    assert summary.loc[("first", "qrc"), "rmse"] == 0.0
    assert summary.loc[("second", "ridge"), "mae"] == 1.0
