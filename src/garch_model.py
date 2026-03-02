from arch import arch_model
import numpy as np

def run_garch(train, test):

    model = arch_model(train, vol="Garch", p=1, q=1)
    res = model.fit(disp="off")

    forecasts = res.forecast(start=len(train)-1, horizon=1)
    pred = forecasts.variance.values[-len(test):, 0]

    return np.sqrt(pred)
