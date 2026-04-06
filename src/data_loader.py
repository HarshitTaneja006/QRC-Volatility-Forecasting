import pandas as pd
import numpy as np
import yfinance as yf
import pandas_datareader.data as web

def load_spy(start, end):
    data = yf.download("SPY", start=start, end=end, progress=False)

    # yfinance may return MultiIndex columns and can omit Adj Close.
    if isinstance(data.columns, pd.MultiIndex):
        data = data.copy()
        data.columns = data.columns.get_level_values(0)

    price_col = "Adj Close" if "Adj Close" in data.columns else "Close"
    if price_col not in data.columns:
        raise KeyError(f"Expected 'Adj Close' or 'Close' in downloaded data columns, got: {list(data.columns)}")

    returns = np.log(data[price_col] / data[price_col].shift(1))
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
