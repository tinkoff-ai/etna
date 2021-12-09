from typing import Dict
from typing import Union

import numpy as np
import pandas as pd
import pytest

from etna.datasets import TSDataset
from etna.transforms import ResampleWithDistributionTransform

DistributionDict = Dict[str, pd.DataFrame]


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


def test_fail_on_incompatible_freq(incompatible_freq_ts):
    resampler = ResampleWithDistributionTransform(
        in_column="exog", inplace=True, distribution_column="target", out_column=None
    )
    with pytest.raises(ValueError, match="Can not infer in_column frequency!"):
        _ = resampler.fit(incompatible_freq_ts.df)


@pytest.mark.parametrize(
    "ts",
    (
        [
            "daily_exog_ts",
            "weekly_exog_same_start_ts",
            "weekly_exog_diff_start_ts",
        ]
    ),
)
def test_fit(ts, request):
    ts = request.getfixturevalue(ts)
    ts, expected_distribution = ts["ts"], ts["distribution"]
    resampler = ResampleWithDistributionTransform(
        in_column="regressor_exog", inplace=True, distribution_column="target", out_column=None
    )
    resampler.fit(ts.df)
    segments = ts.df.columns.get_level_values("segment").unique()
    for segment in segments:
        assert (resampler.segment_transforms[segment].distribution == expected_distribution[segment]).all().all()


@pytest.mark.parametrize(
    "inplace,out_column,expected_resampled_ts",
    (
        [
            (True, None, "inplace_resampled_daily_exog_ts"),
            (False, "resampled_exog", "noninplace_resampled_daily_exog_ts"),
        ]
    ),
)
def test_transform(daily_exog_ts, inplace, out_column, expected_resampled_ts, request):
    daily_exog_ts = daily_exog_ts["ts"]
    expected_resampled_df = request.getfixturevalue(expected_resampled_ts).df
    resampler = ResampleWithDistributionTransform(
        in_column="regressor_exog", inplace=inplace, distribution_column="target", out_column=out_column
    )
    resampled_df = resampler.fit_transform(daily_exog_ts.df)
    assert resampled_df.equals(expected_resampled_df)


@pytest.mark.parametrize(
    "inplace,out_column,expected_resampled_ts",
    (
        [
            (True, None, "inplace_resampled_daily_exog_ts"),
            (False, "resampled_exog", "noninplace_resampled_daily_exog_ts"),
        ]
    ),
)
def test_transform_future(daily_exog_ts, inplace, out_column, expected_resampled_ts, request):
    daily_exog_ts = daily_exog_ts["ts"]
    expected_resampled_ts = request.getfixturevalue(expected_resampled_ts)
    resampler = ResampleWithDistributionTransform(
        in_column="regressor_exog", inplace=inplace, distribution_column="target", out_column=out_column
    )
    daily_exog_ts.fit_transform([resampler])
    future = daily_exog_ts.make_future(3)
    expected_future = expected_resampled_ts.make_future(3)
    assert future.df.equals(expected_future.df)


def test_fit_transform_with_nans(daily_exog_ts_diff_endings):
    resampler = ResampleWithDistributionTransform(
        in_column="regressor_exog", inplace=True, distribution_column="target"
    )
    daily_exog_ts_diff_endings.fit_transform([resampler])
