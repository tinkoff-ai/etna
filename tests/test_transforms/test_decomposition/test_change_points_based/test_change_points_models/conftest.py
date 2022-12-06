import numpy as np
import pandas as pd
import pytest

from etna.datasets import TSDataset
from etna.datasets import generate_ar_df


@pytest.fixture
def ts_with_nans() -> TSDataset:
    """Generate pd.DataFrame with timestamp."""
    df = pd.DataFrame({"timestamp": pd.date_range("2019-12-01", "2019-12-31")})
    tmp = np.zeros(31)
    tmp[8] = None
    df["target"] = tmp
    df["segment"] = "segment_1"
    df = TSDataset.to_dataset(df=df)
    ts = TSDataset(df, freq="H")
    return ts


@pytest.fixture
def simple_ar_df(random_seed):
    df = generate_ar_df(periods=125, start_time="2021-05-20", n_segments=1, ar_coef=[2], freq="D")
    df_ts_format = TSDataset.to_dataset(df)["segment_0"]
    return df_ts_format
