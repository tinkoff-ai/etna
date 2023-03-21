import numpy as np
import pandas as pd
import pytest

from etna.datasets import TSDataset


@pytest.fixture
def simple_ts() -> TSDataset:
    df_1 = pd.DataFrame.from_dict({"timestamp": pd.date_range("2021-06-01", "2021-06-07", freq="D")})
    df_2 = pd.DataFrame.from_dict({"timestamp": pd.date_range("2021-06-01", "2021-06-07", freq="D")})
    df_1["segment"] = "Moscow"
    df_1["target"] = [1.0, 2.0, 3.0, 4.0, 5.0, np.NAN, np.NAN]
    df_1["exog"] = [6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    df_2["segment"] = "Omsk"
    df_2["target"] = [10.0, 20.0, 30.0, 40.0, 50.0, np.NAN, np.NAN]
    df_2["exog"] = [60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0]
    classic_df = pd.concat([df_1, df_2], ignore_index=True)
    df = TSDataset.to_dataset(classic_df)
    ts = TSDataset(df, freq="D")
    return ts


@pytest.fixture
def transformed_simple_df() -> pd.DataFrame:
    df_1 = pd.DataFrame.from_dict({"timestamp": pd.date_range("2021-06-01", "2021-06-07", freq="D")})
    df_2 = pd.DataFrame.from_dict({"timestamp": pd.date_range("2021-06-01", "2021-06-07", freq="D")})
    df_1["segment"] = "Moscow"
    df_1["target"] = [1.0, 2.0, 3.0, 4.0, 5.0, np.NAN, np.NAN]
    df_1["exog"] = [6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    df_1["segment_mean"] = [1, 1.5, 2, 2.5, 3, 3, 3]
    df_2["segment"] = "Omsk"
    df_2["target"] = [10.0, 20.0, 30.0, 40.0, 50.0, np.NAN, np.NAN]
    df_2["exog"] = [60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0]
    df_2["segment_mean"] = [10.0, 15.0, 20.0, 25.0, 30, 30, 30]
    classic_df = pd.concat([df_1, df_2], ignore_index=True)
    df = TSDataset.to_dataset(classic_df)
    return df
