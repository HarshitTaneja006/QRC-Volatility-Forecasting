import numpy as np
import pytest

from src.metrics import diebold_mariano, qlike


def test_qlike_is_zero_for_exact_positive_variance_forecast():
    actual = np.array([0.01, 0.04])

    assert qlike(actual, actual) == pytest.approx(0.0)


def test_diebold_mariano_rejects_single_forecast_comparison():
    with pytest.raises(ValueError, match="at least two"):
        diebold_mariano(np.array([1.0]), np.array([0.9]), np.array([1.1]))


def test_diebold_mariano_returns_finite_values_for_valid_series():
    actual = np.array([1.0, 1.1, 0.9, 1.2, 1.0, 1.05])
    baseline = np.array([1.4, 0.8, 1.3, 0.7, 1.4, 0.6])
    challenger = np.array([1.05, 1.0, 0.95, 1.15, 1.05, 1.0])

    statistic, p_value = diebold_mariano(actual, baseline, challenger)

    assert np.isfinite(statistic)
    assert 0.0 <= p_value <= 1.0
