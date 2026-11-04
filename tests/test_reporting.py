import numpy as np
import pandas as pd

from src.reporting import dm_against_qrc, summarize_forecasts, summarize_magnitude_forecasts


def test_summarize_forecasts_reports_each_model_against_actual():
    forecasts = pd.DataFrame(
        {
            "actual": [0.01, 0.02, 0.03],
            "naive": [0.01, 0.03, 0.02],
            "qrc": [0.012, 0.018, 0.029],
        }
    )

    summary = summarize_forecasts(forecasts)

    assert list(summary.index) == ["naive", "qrc"]
    assert list(summary.columns) == ["rmse", "mae", "qlike"]


def test_magnitude_summary_excludes_variance_specific_qlike():
    forecasts = pd.DataFrame(
        {"actual": [0.1, 0.2], "qrc": [0.11, 0.19], "ridge": [0.12, 0.18]}
    )

    summary = summarize_magnitude_forecasts(forecasts)

    assert list(summary.columns) == ["rmse", "mae"]


def test_dm_against_qrc_marks_degenerate_loss_difference_unavailable():
    forecasts = pd.DataFrame(
        {"actual": [1.0, 2.0, 3.0], "ridge": [1.0, 2.0, 3.0], "qrc": [1.0, 2.0, 3.0]}
    )

    result = dm_against_qrc(forecasts)

    assert np.isnan(result.loc["ridge", "dm_statistic"])
    assert np.isnan(result.loc["ridge", "p_value"])
