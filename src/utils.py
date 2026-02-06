import numpy as np
from scipy import stats

def diebold_mariano_test(real, pred1, pred2, h=1):
    """
    Performs the Diebold-Mariano test to compare two forecast models.
    H0: Both models have the same accuracy.
    If p-value < 0.05, the difference is significant.
    """
    e1 = real - pred1
    e2 = real - pred2
    d = e1**2 - e2**2
    
    mean_d = np.mean(d)
    var_d = np.var(d, ddof=0) / len(d)
    dm_stat = mean_d / np.sqrt(var_d)
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    
    return dm_stat, p_value