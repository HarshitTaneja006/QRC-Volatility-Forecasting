import numpy as np

from src.garch_model import run_garch


def test_garch_returns_one_forecast_for_every_test_observation():
    rng = np.random.default_rng(42)
    predictions = run_garch(rng.normal(size=80), rng.normal(size=6))

    assert predictions.shape == (6,)
    assert np.isfinite(predictions).all()
    assert (predictions > 0).all()
