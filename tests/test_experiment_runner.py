import numpy as np
import pandas as pd

from src.experiment_runner import evaluate_log_variance_models


def test_evaluate_log_variance_models_keeps_common_forecast_index():
    index = pd.bdate_range("2020-01-02", periods=8)
    frame = pd.DataFrame(
        {
            "lag_1": np.linspace(-5.0, -4.3, 8),
            "lag_0": np.linspace(-4.9, -4.2, 8),
            "target": np.linspace(-4.8, -4.1, 8),
        },
        index=index,
    )

    forecasts = evaluate_log_variance_models(
        frame,
        initial_train_size=4,
        refit_every=2,
        ridge_alpha=0.1,
        input_scale=0.5,
        qrc_layers=1,
    )

    assert forecasts.index.equals(index[4:])
    assert list(forecasts.columns) == ["actual", "naive", "ridge", "cosine", "qrc"]
    assert np.isfinite(forecasts.to_numpy()).all()
    assert (forecasts > 0).all().all()


def test_nested_evaluation_returns_aligned_forecasts_and_selection_log():
    from src.experiment_runner import evaluate_nested_log_variance_models

    index = pd.bdate_range("2020-01-02", periods=20)
    frame = pd.DataFrame(
        {
            "lag_1": np.linspace(-5.0, -4.3, 20),
            "lag_0": np.linspace(-4.9, -4.2, 20),
            "target": np.linspace(-4.8, -4.1, 20),
        },
        index=index,
    )
    forecasts, selections = evaluate_nested_log_variance_models(
        frame,
        initial_train_size=10,
        refit_every=4,
        inner_splits=2,
        inner_validation_size=2,
        ridge_candidates=[{"alpha": 0.1}, {"alpha": 1.0}],
        cosine_candidates=[{"alpha": 0.1, "input_scale": 0.5}],
        qrc_candidates=[{"alpha": 0.1, "input_scale": 0.5, "layers": 1}],
    )

    assert forecasts.index.equals(index[10:])
    assert list(forecasts.columns) == ["actual", "naive", "ridge", "cosine", "qrc"]
    assert set(selections) == {"ridge", "cosine", "qrc"}
    assert len(selections["qrc"]) == 3
