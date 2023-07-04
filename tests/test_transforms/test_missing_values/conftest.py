from typing import Dict
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
import pytest

from etna.datasets import TSDataset

frequencies = ["D", "15min"]
DistributionDict = Dict[str, pd.DataFrame]


@pytest.fixture(params=frequencies, ids=frequencies)
def date_range(request) -> pd.DatetimeIndex:
    """Create pd.Series with range of dates."""
    freq = request.param
    dtr = pd.date_range(start="2020-01-01", end="2020-03-01", freq=freq)
    return dtr


@pytest.fixture
def df_all_date_present(date_range: pd.Series) -> pd.DataFrame:
    """Create pd.DataFrame that contains some target on given range of dates without gaps."""
    df = pd.DataFrame({"timestamp": date_range})
    df["target"] = list(range(len(df)))
    df["segment"] = "segment_1"
    df = TSDataset.to_dataset(df)
    df = df.asfreq(date_range.freq)
    return df


@pytest.fixture
def ts_all_date_present_two_segments(df_all_date_present) -> TSDataset:
    """Create TSDataset that contains two segments with some targets on given range of dates without gaps."""
    df_1 = TSDataset.to_flatten(df_all_date_present)
    df_2 = df_1.copy()

    df_1["segment"] = "segment_1"
    df_2["segment"] = "segment_2"

    classic_df = pd.concat([df_1, df_2], ignore_index=True)
    df = TSDataset.to_dataset(classic_df)
    ts = TSDataset(df=df, freq=pd.infer_freq(df.index[-5:]))
    return ts


@pytest.fixture
def ts_with_missing_value_x_index(random_seed, df_all_date_present) -> Tuple[TSDataset, str, int]:
    """Create TSDataset that contains some target on given range of dates with one gap."""
    # index cannot be first or last value,
    # because Imputer should know starting and ending dates
    timestamps = sorted(df_all_date_present.index)[1:-1]
    idx = np.random.choice(timestamps)
    df = df_all_date_present.loc[:, pd.IndexSlice["segment_1", :]]
    df.loc[idx, pd.IndexSlice[:, "target"]] = np.NaN
    ts = TSDataset(df=df, freq=df.index.freqstr)
    return ts, "segment_1", idx


@pytest.fixture
def ts_with_missing_range_x_index(df_all_date_present) -> Tuple[TSDataset, str, list]:
    """Create TSDataset that contains some target on given range of dates with range of gaps."""
    timestamps = sorted(df_all_date_present.index)
    rng = timestamps[2:7]
    df = df_all_date_present.loc[:, pd.IndexSlice["segment_1", :]]
    df.loc[rng, pd.IndexSlice[:, "target"]] = np.NaN
    ts = TSDataset(df=df, freq=df.index.freqstr)
    return ts, "segment_1", rng


@pytest.fixture
def ts_with_missing_range_x_index_two_segments(
    ts_with_missing_range_x_index,
) -> Tuple[TSDataset, list]:
    """Create TSDataset that contains some target on given range of dates with range of gaps."""
    ts_one_segment, _, rng = ts_with_missing_range_x_index
    df_1 = ts_one_segment.to_pandas(flatten=True)
    df_2 = df_1.copy()
    df_1["segment"] = "segment_1"
    df_2["segment"] = "segment_2"
    classic_df = pd.concat([df_1, df_2], ignore_index=True)
    df = TSDataset.to_dataset(classic_df)
    ts = TSDataset(df=df, freq=pd.infer_freq(df.index[-5:]))
    return ts, rng


@pytest.fixture
def df_all_missing(df_all_date_present) -> pd.DataFrame:
    """Create pd.DataFrame with all values set to nan."""
    df_all_date_present.loc[:, :] = np.NaN
    return df_all_date_present


@pytest.fixture
def ts_all_missing_two_segments(ts_all_date_present_two_segments) -> TSDataset:
    """Create TSDataset with all values set to nan."""
    ts_all_date_present_two_segments.loc[:, :] = np.NaN
    return ts_all_date_present_two_segments


@pytest.fixture
def daily_exog_ts() -> Dict[str, Union[TSDataset, DistributionDict]]:
    df1 = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2020-01-05", freq="H", periods=48),
            "segment": "segment_1",
            "target": 1,
        }
    )
    df2 = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2020-01-05", freq="H", periods=48),
            "segment": "segment_2",
            "target": [1] + 23 * [0] + [1] + 23 * [0],
        }
    )
    df = pd.concat([df1, df2], ignore_index=True)

    df_exog1 = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2020-01-05", freq="D", periods=3),
            "segment": "segment_1",
            "regressor_exog": 2,
        }
    )
    df_exog2 = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2020-01-05", freq="D", periods=3),
            "segment": "segment_2",
            "regressor_exog": 40,
        }
    )
    df_exog = pd.concat([df_exog1, df_exog2], ignore_index=True)

    target1 = pd.DataFrame(
        {
            "fold": list(range(24)),
            "distribution": 1 / 24,
        }
    )
    target2 = pd.DataFrame(
        {
            "fold": list(range(24)),
            "distribution": [1] + 23 * [0],
        }
    )

    ts = TSDataset(df=TSDataset.to_dataset(df), freq="H", df_exog=TSDataset.to_dataset(df_exog), known_future="all")
    distribution = {"segment_1": target1, "segment_2": target2}
    return {"ts": ts, "distribution": distribution}


@pytest.fixture()
def daily_exog_ts_diff_endings(daily_exog_ts):
    ts = daily_exog_ts["ts"]
    ts.loc[ts.index[-5] :, pd.IndexSlice["segment_1", "target"]] = np.NAN
    return ts


@pytest.fixture
def inplace_resampled_daily_exog_ts() -> TSDataset:
    df1 = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2020-01-05", freq="H", periods=48),
            "segment": "segment_1",
            "target": 1,
        }
    )
    df2 = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2020-01-05", freq="H", periods=48),
            "segment": "segment_2",
            "target": [1] + 23 * [0] + [1] + 23 * [0],
        }
    )
    df = pd.concat([df1, df2], ignore_index=True)

    df_exog1 = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2020-01-05", freq="H", periods=72),
            "segment": "segment_1",
            "regressor_exog": 2 / 24,
        }
    )
    df_exog2 = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2020-01-05", freq="H", periods=72),
            "segment": "segment_2",
            "regressor_exog": [40] + 23 * [0] + [40] + 23 * [0] + [40] + 23 * [0],
        }
    )
    df_exog = pd.concat([df_exog1, df_exog2], ignore_index=True)

    ts = TSDataset(df=TSDataset.to_dataset(df), freq="H", df_exog=TSDataset.to_dataset(df_exog), known_future="all")
    return ts


@pytest.fixture
def noninplace_resampled_daily_exog_ts() -> TSDataset:
    df1 = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2020-01-05", freq="H", periods=48),
            "segment": "segment_1",
            "target": 1,
        }
    )
    df2 = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2020-01-05", freq="H", periods=48),
            "segment": "segment_2",
            "target": [1] + 23 * [0] + [1] + 23 * [0],
        }
    )
    df = pd.concat([df1, df2], ignore_index=True)

    df_exog1 = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2020-01-05", freq="H", periods=72),
            "segment": "segment_1",
            "regressor_exog": [2] + 23 * [np.NAN] + [2] + 23 * [np.NAN] + [2] + 23 * [np.NAN],
            "resampled_exog": 2 / 24,
        }
    )
    df_exog2 = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2020-01-05", freq="H", periods=72),
            "segment": "segment_2",
            "regressor_exog": [40] + 23 * [np.NAN] + [40] + 23 * [np.NAN] + [40] + 23 * [np.NAN],
            "resampled_exog": [40] + 23 * [0] + [40] + 23 * [0] + [40] + 23 * [0],
        }
    )
    df_exog = pd.concat([df_exog1, df_exog2], ignore_index=True)

    ts = TSDataset(df=TSDataset.to_dataset(df), freq="H", df_exog=TSDataset.to_dataset(df_exog), known_future="all")
    return ts


@pytest.fixture
def weekly_exog_same_start_ts() -> Dict[str, Union[TSDataset, DistributionDict]]:
    """Target and exog columns start on Monday."""
    df1 = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2020-01-05", freq="D", periods=14),
            "segment": "segment_1",
            "target": 1,
        }
    )
    df2 = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2020-01-05", freq="D", periods=14),
            "segment": "segment_2",
            "target": [1] + 6 * [0] + [1] + 6 * [0],
        }
    )
    df = pd.concat([df1, df2], ignore_index=True)

    df_exog1 = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2020-01-05", freq="W", periods=3),
            "segment": "segment_1",
            "regressor_exog": 2,
        }
    )
    df_exog2 = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2020-01-05", freq="W", periods=3),
            "segment": "segment_2",
            "regressor_exog": 40,
        }
    )
    df_exog = pd.concat([df_exog1, df_exog2], ignore_index=True)

    target1 = pd.DataFrame(
        {
            "fold": list(range(7)),
            "distribution": 1 / 7,
        }
    )
    target2 = pd.DataFrame(
        {
            "fold": list(range(7)),
            "distribution": [1] + 6 * [0],
        }
    )
    distribution = {"segment_1": target1, "segment_2": target2}
    ts = TSDataset(df=TSDataset.to_dataset(df), freq="D", df_exog=TSDataset.to_dataset(df_exog), known_future="all")

    return {"ts": ts, "distribution": distribution}


@pytest.fixture
def weekly_exog_diff_start_ts() -> Dict[str, Union[TSDataset, DistributionDict]]:
    """Target starts on Thursday and exog starts on Monday."""
    df1 = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2020-01-08", freq="D", periods=14),
            "segment": "segment_1",
            "target": 1,
        }
    )
    df2 = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2020-01-08", freq="D", periods=14),
            "segment": "segment_2",
            "target": [1] + 6 * [0] + [1] + 6 * [0],
        }
    )
    df = pd.concat([df1, df2], ignore_index=True)

    df_exog1 = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2020-01-05", freq="W", periods=4),
            "segment": "segment_1",
            "regressor_exog": 2,
        }
    )
    df_exog2 = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2020-01-05", freq="W", periods=4),
            "segment": "segment_2",
            "regressor_exog": 40,
        }
    )
    df_exog = pd.concat([df_exog1, df_exog2], ignore_index=True)

    target1 = pd.DataFrame(
        {
            "fold": list(range(7)),
            "distribution": 1 / 7,
        }
    )
    target2 = pd.DataFrame(
        {
            "fold": list(range(7)),
            "distribution": [0, 0, 0, 1, 0, 0, 0],
        }
    )

    ts = TSDataset(df=TSDataset.to_dataset(df), freq="D", df_exog=TSDataset.to_dataset(df_exog), known_future="all")
    distribution = {"segment_1": target1, "segment_2": target2}
    return {"ts": ts, "distribution": distribution}


@pytest.fixture
def incompatible_freq_ts() -> TSDataset:
    df1 = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2020-01-01", freq="7T", periods=20),
            "segment": "segment_1",
            "target": 1,
        }
    )
    df2 = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2020-01-01", freq="7T", periods=20),
            "segment": "segment_2",
            "target": 2,
        }
    )
    df = pd.concat([df1, df2], ignore_index=True)

    df_exog1 = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2020-01-01", freq="H", periods=3),
            "segment": "segment_1",
            "exog": 2,
        }
    )
    df_exog2 = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2020-01-01", freq="H", periods=3),
            "segment": "segment_2",
            "exog": 40,
        }
    )
    df_exog = pd.concat([df_exog1, df_exog2], ignore_index=True)

    ts = TSDataset(df=TSDataset.to_dataset(df), freq="7T", df_exog=TSDataset.to_dataset(df_exog))
    return ts
