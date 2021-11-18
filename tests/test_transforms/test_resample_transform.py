from typing import Dict

import numpy as np
import pandas as pd
import pytest

from etna.datasets import TSDataset
from etna.transforms import ResampleWithDistributionTransform


@pytest.fixture
def daily_exog_df() -> pd.DataFrame:
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
            "exog": 2,
        }
    )
    df_exog2 = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2020-01-05", freq="D", periods=3),
            "segment": "segment_2",
            "exog": 40,
        }
    )
    df_exog = pd.concat([df_exog1, df_exog2], ignore_index=True)

    ts = TSDataset(df=TSDataset.to_dataset(df), freq="H", df_exog=TSDataset.to_dataset(df_exog))
    return ts.df


@pytest.fixture
def inplace_resampled_daily_exog_df() -> pd.DataFrame:
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
            "exog": 2 / 24,
        }
    )
    df_exog2 = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2020-01-05", freq="H", periods=72),
            "segment": "segment_2",
            "exog": [40] + 23 * [0] + [40] + 23 * [0] + [40] + 23 * [0],
        }
    )
    df_exog = pd.concat([df_exog1, df_exog2], ignore_index=True)

    ts = TSDataset(df=TSDataset.to_dataset(df), freq="H", df_exog=TSDataset.to_dataset(df_exog))
    return ts.df


@pytest.fixture
def noninplace_resampled_daily_exog_df() -> pd.DataFrame:
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
            "exog": [2] + 23 * [np.NAN] + [2] + 23 * [np.NAN] + [2] + 23 * [np.NAN],
            "resampled_exog": 2 / 24,
        }
    )
    df_exog2 = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2020-01-05", freq="H", periods=72),
            "segment": "segment_2",
            "exog": [40] + 23 * [np.NAN] + [40] + 23 * [np.NAN] + [40] + 23 * [np.NAN],
            "resampled_exog": [40] + 23 * [0] + [40] + 23 * [0] + [40] + 23 * [0],
        }
    )
    df_exog = pd.concat([df_exog1, df_exog2], ignore_index=True)

    ts = TSDataset(df=TSDataset.to_dataset(df), freq="H", df_exog=TSDataset.to_dataset(df_exog))
    return ts.df


@pytest.fixture
def weekly_exog_same_start_df() -> pd.DataFrame:
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
            "exog": 2,
        }
    )
    df_exog2 = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2020-01-05", freq="W", periods=3),
            "segment": "segment_2",
            "exog": 40,
        }
    )
    df_exog = pd.concat([df_exog1, df_exog2], ignore_index=True)

    ts = TSDataset(df=TSDataset.to_dataset(df), freq="D", df_exog=TSDataset.to_dataset(df_exog))
    return ts.df


@pytest.fixture
def weekly_exog_diff_start_df() -> pd.DataFrame:
    """Target starts on Thursday and exog starts on the next Monday."""
    df1 = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2020-01-01", freq="D", periods=14),
            "segment": "segment_1",
            "target": 1,
        }
    )
    df2 = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2020-01-01", freq="D", periods=14),
            "segment": "segment_2",
            "target": [1] + 6 * [0] + [1] + 6 * [0],
        }
    )
    df = pd.concat([df1, df2], ignore_index=True)

    df_exog1 = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2020-01-05", freq="W", periods=3),
            "segment": "segment_1",
            "exog": 2,
        }
    )
    df_exog2 = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2020-01-05", freq="W", periods=3),
            "segment": "segment_2",
            "exog": 40,
        }
    )
    df_exog = pd.concat([df_exog1, df_exog2], ignore_index=True)

    ts = TSDataset(df=TSDataset.to_dataset(df), freq="D", df_exog=TSDataset.to_dataset(df_exog))
    return ts.df


@pytest.fixture
def incompatible_freq_df():
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
    return ts.df


@pytest.fixture
def distribution_daily_dict() -> Dict[str, pd.DataFrame]:
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
    distribution = {"segment_1": target1, "segment_2": target2}
    return distribution


@pytest.fixture
def distribution_weekly_same_start_dict() -> Dict[str, pd.DataFrame]:
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
    return distribution


@pytest.fixture
def distribution_weekly_diff_start_dict() -> Dict[str, pd.DataFrame]:
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
    distribution = {"segment_1": target1, "segment_2": target2}
    return distribution


def test_fail_on_incompatible_freq(incompatible_freq_df):
    resampler = ResampleWithDistributionTransform(
        in_column="exog", inplace=True, distribution_column="target", out_column=None
    )
    with pytest.raises(
        ValueError,
        match="Can not infer in_column frequency!"
        "Check that in_column frequency is compatible with dataset frequency.",
    ):
        _ = resampler.fit(incompatible_freq_df)


@pytest.mark.parametrize(
    "df,expected_distribution",
    (
        [
            ("daily_exog_df", "distribution_daily_dict"),
            ("weekly_exog_same_start_df", "distribution_weekly_same_start_dict"),
            ("weekly_exog_diff_start_df", "distribution_weekly_diff_start_dict"),
        ]
    ),
)
def test_fit(df, expected_distribution, request):
    df, expected_distribution = request.getfixturevalue(df), request.getfixturevalue(expected_distribution)
    resampler = ResampleWithDistributionTransform(
        in_column="exog", inplace=True, distribution_column="target", out_column=None
    )
    resampler.fit(df)
    segments = df.columns.get_level_values("segment").unique()
    for segment in segments:
        assert (resampler.segment_transforms[segment].distribution == expected_distribution[segment]).all().all()


@pytest.mark.parametrize(
    "inplace,out_column,expected_resampled_df",
    (
        [
            (True, None, "inplace_resampled_daily_exog_df"),
            (False, "resampled_exog", "noninplace_resampled_daily_exog_df"),
        ]
    ),
)
def test_transform(daily_exog_df, inplace, out_column, expected_resampled_df, request):
    expected_resampled_df = request.getfixturevalue(expected_resampled_df)
    resampler = ResampleWithDistributionTransform(
        in_column="exog", inplace=inplace, distribution_column="target", out_column=out_column
    )
    resampled_df = resampler.fit_transform(daily_exog_df)
    assert resampled_df.equals(expected_resampled_df)
