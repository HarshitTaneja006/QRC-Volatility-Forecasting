import numpy as np

class GARCHModel:
    """Placeholder for GARCH(1,1) model baseline."""

    def __init__(self, omega=0.0001, alpha=0.1, beta=0.85):
        self.omega = omega
        self.alpha = alpha
        self.beta = beta

    def forecast(self, returns):
        n = len(returns)
        sigma2 = np.zeros(n)
        sigma2[0] = np.var(returns)

        for t in range(1, n):
            sigma2[t] = (self.omega
                         + self.alpha * returns[t - 1] ** 2
                         + self.beta * sigma2[t - 1])

        return np.sqrt(sigma2)
