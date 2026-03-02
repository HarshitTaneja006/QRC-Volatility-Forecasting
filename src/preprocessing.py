from sklearn.preprocessing import StandardScaler
import numpy as np

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
