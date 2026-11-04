from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

def normalize_series(series, split):
    scaler = StandardScaler()
    train = series[:split]
    test = series[split:]

    train_scaled = scaler.fit_transform(train.values.reshape(-1,1)).flatten()
    test_scaled = scaler.transform(test.values.reshape(-1,1)).flatten()

    return train_scaled, test_scaled

def create_windows(data, window):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window])
        y.append(abs(data[i+window]))
    return np.array(X), np.array(y)


def make_lagged_supervised(feature, target, lookback):
    """Create dated lag features without crossing the target's forecast boundary."""
    if lookback < 1:
        raise ValueError("lookback must be at least one")
    feature = pd.Series(feature, dtype=float)
    target = pd.Series(target, dtype=float)
    if not feature.index.equals(target.index):
        raise ValueError("feature and target must have identical date indexes")

    columns = {
        f"lag_{lag}": feature.shift(lag)
        for lag in range(lookback - 1, -1, -1)
    }
    frame = pd.DataFrame(columns, index=feature.index)
    frame["target"] = target
    return frame.dropna()
