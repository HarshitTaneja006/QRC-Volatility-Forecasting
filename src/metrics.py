import numpy as np
from scipy import stats

def rmse(y, yhat):
    return np.sqrt(np.mean((y - yhat)**2))

def directional_accuracy(y, yhat):
    return np.mean(np.sign(y) == np.sign(yhat))

def diebold_mariano(y, pred1, pred2):

    e1 = y - pred1
    e2 = y - pred2
    d = e1**2 - e2**2

    mean_d = np.mean(d)
    var_d = np.var(d) / len(d)

    dm = mean_d / np.sqrt(var_d)
    p = 2 * (1 - stats.norm.cdf(abs(dm)))

    return dm, p
