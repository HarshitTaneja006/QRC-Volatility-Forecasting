from arch import arch_model
import numpy as np
import pandas as pd

def run_garch(train, test):
    """Produce a recursively projected conditional-variance forecast per test date."""
    train = np.asarray(train, dtype=float)
    test = np.asarray(test, dtype=float)
    if len(train) < 10:
        raise ValueError("GARCH requires at least ten training observations")
    if not len(test):
        raise ValueError("test data must not be empty")

    # `arch` is numerically better conditioned when return units are percentages.
    scale = 100.0
    model = arch_model(
        train * scale, mean="Zero", vol="Garch", p=1, q=1, dist="t", rescale=False
    )
    res = model.fit(disp="off")
    variance = res.forecast(horizon=len(test), reindex=False).variance.values[-1]
    return np.sqrt(np.maximum(variance, np.finfo(float).tiny)) / scale


def blockwise_garch_variance_forecast(
    returns, forecast_index, *, refit_every, forecast_function=run_garch
):
    """Refit GARCH before each forecast block and return aligned variance forecasts."""
    returns = pd.Series(returns, dtype=float).sort_index()
    forecast_index = pd.Index(forecast_index)
    if refit_every < 1:
        raise ValueError("refit_every must be at least one")
    if not forecast_index.isin(returns.index).all():
        raise ValueError("every forecast index must occur in returns")

    predictions = []
    for block_start in range(0, len(forecast_index), refit_every):
        block_index = forecast_index[block_start : block_start + refit_every]
        train_end = block_index[0]
        train = returns.loc[:train_end].to_numpy()
        standard_deviation = forecast_function(train, np.empty(len(block_index)))
        if len(standard_deviation) != len(block_index):
            raise ValueError("GARCH forecast length does not match forecast block")
        predictions.extend(np.square(standard_deviation))
    return pd.Series(predictions, index=forecast_index, name="garch")
