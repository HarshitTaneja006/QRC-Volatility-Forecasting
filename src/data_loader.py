import pandas as pd
import numpy as np
import yfinance as yf
import pandas_datareader.data as web

def load_spy(start, end):
    data = yf.download("SPY", start=start, end=end, progress=False)
    returns = np.log(data["Adj Close"] / data["Adj Close"].shift(1))
    return returns.dropna()

def load_gdp(start, end):
    gdp = web.DataReader("GDPC1", "fred", start, end)
    # EXACT equation from paper:
    gdp["growth"] = 100 * np.log(gdp["GDPC1"] / gdp["GDPC1"].shift(4))
    return gdp["growth"].dropna()

def load_indpro(start, end):
    ip = web.DataReader("INDPRO", "fred", start, end)
    ip["growth"] = 100 * np.log(ip["INDPRO"] / ip["INDPRO"].shift(1))
    return ip["growth"].dropna()
