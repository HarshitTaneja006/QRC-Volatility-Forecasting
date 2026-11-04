import numpy as np
import pandas as pd

from src.garch_model import blockwise_garch_variance_forecast


def test_blockwise_garch_forecast_keeps_requested_dates_and_block_lengths():
    index = pd.bdate_range("2020-01-02", periods=8)
    returns = pd.Series(np.linspace(0.01, 0.08, 8), index=index)
    requested = index[3:]
    calls = []

    def fake_garch(train, test, **_kwargs):
        calls.append((len(train), len(test)))
        return np.repeat(0.2, len(test))

    result = blockwise_garch_variance_forecast(
        returns, requested, refit_every=2, forecast_function=fake_garch
    )

    assert result.index.equals(requested)
    np.testing.assert_allclose(result.to_numpy(), [0.04] * 5)
    assert calls == [(4, 2), (6, 2), (8, 1)]
