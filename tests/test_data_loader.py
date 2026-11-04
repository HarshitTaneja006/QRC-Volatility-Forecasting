import pandas as pd

from src.data_loader import load_spy_ohlc


def test_load_spy_ohlc_flattens_yfinance_multiindex_columns():
    index = pd.bdate_range("2020-01-02", periods=2)
    columns = pd.MultiIndex.from_product(
        [["Open", "High", "Low", "Close", "Volume"], ["SPY"]]
    )
    raw = pd.DataFrame(
        [[100, 105, 99, 104, 1000], [104, 106, 102, 103, 1200]],
        index=index,
        columns=columns,
    )

    result = load_spy_ohlc(
        "2020-01-01", "2020-01-04", downloader=lambda *_args, **_kwargs: raw
    )

    assert list(result.columns) == ["Open", "High", "Low", "Close"]
    assert result.index.equals(index)


def test_load_ohlc_accepts_a_non_spy_ticker():
    observed = []

    def downloader(ticker, **kwargs):
        observed.append(ticker)
        return pd.DataFrame(
            {"Open": [1.0], "High": [2.0], "Low": [0.5], "Close": [1.5]},
            index=pd.to_datetime(["2020-01-01"]),
        )

    result = load_spy_ohlc("2020-01-01", "2020-01-02", ticker="EFA", downloader=downloader)

    assert observed == ["EFA"]
    assert result.loc[pd.Timestamp("2020-01-01"), "Close"] == 1.5
