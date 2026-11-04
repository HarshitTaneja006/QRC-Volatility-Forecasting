import numpy as np
import pandas as pd

from src.preprocessing import make_lagged_supervised


def test_lagged_supervised_uses_only_information_available_at_feature_date():
    index = pd.bdate_range("2020-01-02", periods=5)
    feature = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=index)
    target = pd.Series([2.0, 3.0, 4.0, 5.0, np.nan], index=index)

    frame = make_lagged_supervised(feature, target, lookback=3)

    assert list(frame.index) == [index[2], index[3]]
    assert frame.loc[index[2], ["lag_2", "lag_1", "lag_0"]].tolist() == [1.0, 2.0, 3.0]
    assert frame.loc[index[2], "target"] == 4.0
