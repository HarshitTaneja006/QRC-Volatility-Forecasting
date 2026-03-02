import numpy as np
from scipy import stats

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

def directional_accuracy(y_true, y_pred):
    correct = 0
    for i in range(1, len(y_true)):
        if (y_true[i] - y_true[i-1]) * (y_pred[i] - y_pred[i-1]) > 0:
            correct += 1
    return correct / (len(y_true) - 1)

def diebold_mariano(y_true, pred1, pred2):
    e1 = y_true - pred1
    e2 = y_true - pred2
    d = e1**2 - e2**2

    mean_d = np.mean(d)
    var_d = np.var(d, ddof=1) / len(d)

    dm_stat = mean_d / np.sqrt(var_d)
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))

    return dm_stat, p_value
