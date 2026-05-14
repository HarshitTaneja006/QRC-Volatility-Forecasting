import pandas as pd
import numpy as np
import yfinance as yf


def _load_fred_series(series_id, start, end):
    """Load a FRED series without pandas_datareader (Python 3.13 compatible)."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    data = pd.read_csv(url, encoding="utf-8-sig")
    data.columns = [str(col).strip().lstrip("\ufeff") for col in data.columns]

    date_col = next(
        (
            col
            for col in data.columns
            if col.upper() in {"DATE", "OBSERVATION_DATE"}
        ),
        None,
    )
    value_col = series_id if series_id in data.columns else None

    if value_col is None:
        value_col = next((col for col in data.columns if col.upper() == "VALUE"), None)

    if date_col is None or value_col is None:
        raise ValueError(
            f"Unexpected FRED response format for {series_id}. "
            f"Columns received: {list(data.columns)}"
        )

    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    data[value_col] = pd.to_numeric(data[value_col], errors="coerce")
    data = data.dropna(subset=[date_col, value_col])

    series = data.set_index(date_col)[value_col]
    series = pd.to_numeric(series, errors="coerce").dropna()

    if start is not None:
        series = series[series.index >= pd.to_datetime(start)]
    if end is not None:
        series = series[series.index <= pd.to_datetime(end)]

    return series.sort_index()

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
    gdp = pd.DataFrame({"GDPC1": _load_fred_series("GDPC1", start, end)})
    # EXACT equation from paper:
    gdp["growth"] = 100 * np.log(gdp["GDPC1"] / gdp["GDPC1"].shift(4))
    return gdp["growth"].dropna()

def load_indpro(start, end):
    ip = pd.DataFrame({"INDPRO": _load_fred_series("INDPRO", start, end)})
    ip["growth"] = 100 * np.log(ip["INDPRO"] / ip["INDPRO"].shift(1))
    return ip["growth"].dropna()
