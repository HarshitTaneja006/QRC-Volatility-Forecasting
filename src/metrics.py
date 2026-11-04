import numpy as np
from scipy import stats


def _as_aligned_arrays(*arrays):
    converted = tuple(np.asarray(array, dtype=float).reshape(-1) for array in arrays)
    lengths = {len(array) for array in converted}
    if len(lengths) != 1:
        raise ValueError("all forecast arrays must have identical lengths")
    if not converted[0].size:
        raise ValueError("forecast arrays must not be empty")
    if not all(np.isfinite(array).all() for array in converted):
        raise ValueError("forecast arrays must contain only finite values")
    return converted


def rmse(y, yhat):
    y, yhat = _as_aligned_arrays(y, yhat)
    return np.sqrt(np.mean((y - yhat) ** 2))


def mae(y, yhat):
    y, yhat = _as_aligned_arrays(y, yhat)
    return np.mean(np.abs(y - yhat))


def qlike(realized_variance, forecast_variance):
    """Mean QLIKE loss for strictly positive realized and forecast variances."""
    realized_variance, forecast_variance = _as_aligned_arrays(
        realized_variance, forecast_variance
    )
    if (realized_variance <= 0).any() or (forecast_variance <= 0).any():
        raise ValueError("QLIKE requires strictly positive variances")
    ratio = realized_variance / forecast_variance
    return np.mean(ratio - np.log(ratio) - 1.0)

def directional_accuracy(y, yhat):
    y, yhat = _as_aligned_arrays(y, yhat)
    if len(y) < 2:
        raise ValueError("directional accuracy requires at least two observations")
    return np.mean(np.sign(np.diff(y)) == np.sign(np.diff(yhat)))


def diebold_mariano(y, pred1, pred2, *, max_lag=None):
    """Two-sided Diebold-Mariano test with Bartlett HAC long-run variance.

    Positive statistics mean ``pred2`` has lower squared-error loss than ``pred1``.
    """
    y, pred1, pred2 = _as_aligned_arrays(y, pred1, pred2)
    n_obs = len(y)
    if n_obs < 2:
        raise ValueError("Diebold-Mariano test requires at least two observations")

    differential = (y - pred1) ** 2 - (y - pred2) ** 2
    centered = differential - differential.mean()
    if max_lag is None:
        max_lag = min(n_obs - 1, int(np.floor(4 * (n_obs / 100) ** (2 / 9))))
    if max_lag < 0 or max_lag >= n_obs:
        raise ValueError("max_lag must be between 0 and len(y) - 1")

    long_run_variance = np.mean(centered * centered)
    for lag in range(1, max_lag + 1):
        autocovariance = np.mean(centered[lag:] * centered[:-lag])
        long_run_variance += 2 * (1 - lag / (max_lag + 1)) * autocovariance
    if long_run_variance <= np.finfo(float).eps:
        raise ValueError("loss differential has zero long-run variance")

    statistic = differential.mean() / np.sqrt(long_run_variance / n_obs)
    p_value = 2 * stats.norm.sf(abs(statistic))
    return float(statistic), float(p_value)
