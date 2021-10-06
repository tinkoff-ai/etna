from typing import Tuple

import numpy as np
import pandas as pd
import pytest

from etna.datasets import TSDataset
from etna.models import NaiveModel
from etna.transforms.imputation import TimeSeriesImputerTransform
from etna.transforms.imputation import _OneSegmentTimeSeriesImputerTransform

frequencies = ["D", "15min"]


@pytest.fixture(params=frequencies, ids=frequencies)
def date_range(request) -> pd.Series:
    """Create pd.Series with range of dates."""
    freq = request.param
    dtr = pd.date_range(start="2020-01-01", end="2020-03-01", freq=freq)
    return dtr


@pytest.fixture()
def all_date_present_df(date_range: pd.Series) -> pd.DataFrame:
    """Create pd.DataFrame that contains some target on given range of dates without gaps."""
    df = pd.DataFrame({"timestamp": date_range})
    df["target"] = [i for i in range(len(df))]
    df.set_index("timestamp", inplace=True)
    return df


@pytest.fixture()
def all_date_present_df_two_segments(all_date_present_df: pd.Series) -> pd.DataFrame:
    """Create pd.DataFrame that contains two segments with some targets on given range of dates without gaps."""
    df_1 = all_date_present_df.reset_index()
    df_2 = all_date_present_df.copy().reset_index()

    df_1["segment"] = "segment_1"
    df_2["segment"] = "segment_2"

    classic_df = pd.concat([df_1, df_2], ignore_index=True)
    df = TSDataset.to_dataset(classic_df)
    return df


def test_wrong_init_one_segment():
    """Check that imputer for one segment fails to init with wrong imputing strategy."""
    with pytest.raises(ValueError):
        _ = _OneSegmentTimeSeriesImputerTransform(strategy="wrong_strategy")


def test_wrong_init_two_segments(all_date_present_df_two_segments):
    """Check that imputer for two segments fails to fit_transform with wrong imputing strategy."""
    with pytest.raises(ValueError):
        _ = TimeSeriesImputerTransform(strategy="wrong_strategy")


@pytest.fixture()
def df_with_missing_value_x_index(all_date_present_df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Create pd.DataFrame that contains some target on given range of dates with one gap."""
    # index cannot be first or last value,
    # because Imputer should know starting and ending dates
    timestamps = sorted(all_date_present_df.index)[1:-1]
    idx = np.random.choice(timestamps)
    df = all_date_present_df
    df.loc[idx, "target"] = np.NaN
    return df, idx


@pytest.fixture()
def df_with_missing_range_x_index(all_date_present_df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
    """Create pd.DataFrame that contains some target on given range of dates with range of gaps."""
    timestamps = sorted(all_date_present_df.index)
    rng = timestamps[2:7]
    df = all_date_present_df
    df.loc[rng, "target"] = np.NaN
    return df, rng


@pytest.fixture()
def df_with_missing_range_x_index_two_segments(
    df_with_missing_range_x_index: pd.DataFrame
) -> Tuple[pd.DataFrame, list]:
    """Create pd.DataFrame that contains some target on given range of dates with range of gaps."""
    df_one_segment, rng = df_with_missing_range_x_index
    df_1 = df_one_segment.reset_index()
    df_2 = df_one_segment.copy().reset_index()
    df_1["segment"] = "segment_1"
    df_2["segment"] = "segment_2"
    classic_df = pd.concat([df_1, df_2], ignore_index=True)
    df = TSDataset.to_dataset(classic_df)
    return df, rng


@pytest.mark.smoke
@pytest.mark.parametrize("fill_strategy", ["mean", "zero", "running_mean", "forward_fill"])
def test_all_dates_present_impute(all_date_present_df: pd.DataFrame, fill_strategy: str):
    """Check that imputer does nothing with series without gaps."""
    imputer = _OneSegmentTimeSeriesImputerTransform(strategy=fill_strategy)
    result = imputer.fit_transform(all_date_present_df)
    np.testing.assert_array_equal(all_date_present_df["target"], result["target"])


@pytest.mark.smoke
@pytest.mark.parametrize("fill_strategy", ["mean", "zero", "running_mean", "forward_fill"])
def test_all_dates_present_impute_two_segments(all_date_present_df_two_segments: pd.DataFrame, fill_strategy: str):
    """Check that imputer does nothing with series without gaps."""
    imputer = TimeSeriesImputerTransform(strategy=fill_strategy)
    result = imputer.fit_transform(all_date_present_df_two_segments)
    for segment in result.columns.get_level_values("segment"):
        np.testing.assert_array_equal(all_date_present_df_two_segments[segment]["target"], result[segment]["target"])


def test_one_missing_value_zero(df_with_missing_value_x_index: pd.DataFrame):
    """Check that imputer with zero-strategy works correctly in case of one missing value in data."""
    df, idx = df_with_missing_value_x_index
    imputer = _OneSegmentTimeSeriesImputerTransform(strategy="zero")
    result = imputer.fit_transform(df)["target"]
    assert result.loc[idx] == 0
    assert not result.isna().any()


def test_range_missing_zero(df_with_missing_range_x_index: pd.DataFrame):
    """Check that imputer with zero-strategy works correctly in case of range of missing values in data."""
    df, rng = df_with_missing_range_x_index
    imputer = _OneSegmentTimeSeriesImputerTransform(strategy="zero")
    result = imputer.fit_transform(df)["target"]
    expected_series = pd.Series(index=rng, data=[0 for _ in rng], name="target")
    np.testing.assert_array_almost_equal(result.loc[rng].reset_index(drop=True), expected_series)
    assert not result.isna().any()


def test_one_missing_value_mean(df_with_missing_value_x_index: pd.DataFrame):
    """Check that imputer with mean-strategy works correctly in case of one missing value in data."""
    df, idx = df_with_missing_value_x_index
    imputer = _OneSegmentTimeSeriesImputerTransform(strategy="mean")
    expected_value = df["target"].mean()
    result = imputer.fit_transform(df)["target"]
    assert result.loc[idx] == expected_value
    assert not result.isna().any()


def test_range_missing_mean(df_with_missing_range_x_index):
    """Check that imputer with mean-strategy works correctly in case of range of missing values in data."""
    df, rng = df_with_missing_range_x_index
    imputer = _OneSegmentTimeSeriesImputerTransform(strategy="mean")
    result = imputer.fit_transform(df)["target"]
    expected_value = df["target"].mean()
    expected_series = pd.Series(index=rng, data=[expected_value for _ in rng], name="target")
    np.testing.assert_array_almost_equal(result.loc[rng].reset_index(drop=True), expected_series)
    assert not result.isna().any()


def test_one_missing_value_forward_fill(df_with_missing_value_x_index):
    """Check that imputer with forward-fill-strategy works correctly in case of one missing value in data."""
    df, idx = df_with_missing_value_x_index
    imputer = _OneSegmentTimeSeriesImputerTransform(strategy="forward_fill")
    result = imputer.fit_transform(df)["target"]

    timestamps = np.array(sorted(df.index))
    timestamp_idx = np.where(timestamps == idx)[0][0]
    expected_value = df.loc[timestamps[timestamp_idx - 1], "target"]
    assert result.loc[idx] == expected_value
    assert not result.isna().any()


def test_range_missing_forward_fill(df_with_missing_range_x_index: pd.DataFrame):
    """Check that imputer with forward-fill-strategy works correctly in case of range of missing values in data."""
    df, rng = df_with_missing_range_x_index
    imputer = _OneSegmentTimeSeriesImputerTransform(strategy="forward_fill")
    result = imputer.fit_transform(df)["target"]

    timestamps = np.array(sorted(df.index))
    rng = [pd.Timestamp(x) for x in rng]
    timestamp_idx = min(np.where([x in rng for x in timestamps])[0])
    expected_value = df.loc[timestamps[timestamp_idx - 1], "target"]
    expected_series = pd.Series(index=rng, data=[expected_value for _ in rng], name="target")
    np.testing.assert_array_almost_equal(result.loc[rng], expected_series)
    assert not result.isna().any()


@pytest.mark.parametrize("window", [1, -1, 2])
def test_one_missing_value_running_mean(df_with_missing_value_x_index: pd.DataFrame, window: int):
    """Check that imputer with running-mean-strategy works correctly in case of one missing value in data."""
    df, idx = df_with_missing_value_x_index
    timestamps = np.array(sorted(df.index))
    timestamp_idx = np.where(timestamps == idx)[0][0]
    imputer = _OneSegmentTimeSeriesImputerTransform(strategy="running_mean", window=window)
    if window == -1:
        expected_value = df.loc[: timestamps[timestamp_idx - 1], "target"].mean()
    else:
        expected_value = df.loc[timestamps[timestamp_idx - window] : timestamps[timestamp_idx - 1], "target"].mean()
    result = imputer.fit_transform(df)["target"]
    assert result.loc[idx] == expected_value
    assert not result.isna().any()


@pytest.mark.parametrize("window", [1, -1, 2])
def test_range_missing_running_mean(df_with_missing_range_x_index: pd.DataFrame, window: int):
    """Check that imputer with running-mean-strategy works correctly in case of range of missing values in data."""
    df, rng = df_with_missing_range_x_index
    timestamps = np.array(sorted(df.index))
    timestamp_idxs = np.where([x in rng for x in timestamps])[0]
    imputer = _OneSegmentTimeSeriesImputerTransform(strategy="running_mean", window=window)
    result = imputer.fit_transform(df)["target"]

    assert not result.isna().any()
    for idx in timestamp_idxs:
        if window == -1:
            expected_value = result.loc[: timestamps[idx - 1]].mean()
        else:
            expected_value = result.loc[timestamps[idx - window] : timestamps[idx - 1]].mean()
        assert result.loc[timestamps[idx]] == expected_value


@pytest.mark.parametrize("fill_strategy", ["mean", "zero", "running_mean", "forward_fill"])
def test_inverse_transform_one_segment(df_with_missing_range_x_index: pd.DataFrame, fill_strategy: str):
    """Check that transform + inverse_transform don't change original df for one segment."""
    df, rng = df_with_missing_range_x_index
    imputer = _OneSegmentTimeSeriesImputerTransform(strategy=fill_strategy)
    transform_result = imputer.fit_transform(df)
    inverse_transform_result = imputer.inverse_transform(transform_result)
    np.testing.assert_array_equal(df, inverse_transform_result)


@pytest.mark.parametrize("fill_strategy", ["mean", "zero", "running_mean", "forward_fill"])
def test_inverse_transform_many_segments(df_with_missing_range_x_index_two_segments: pd.DataFrame, fill_strategy: str):
    """Check that transform + inverse_transform don't change original df for two segments."""
    df, rng = df_with_missing_range_x_index_two_segments
    imputer = TimeSeriesImputerTransform(strategy=fill_strategy)
    transform_result = imputer.fit_transform(df)
    inverse_transform_result = imputer.inverse_transform(transform_result)
    np.testing.assert_array_equal(df, inverse_transform_result)


@pytest.mark.parametrize("fill_strategy", ["mean", "zero", "running_mean", "forward_fill"])
def test_inverse_transform_in_forecast(df_with_missing_range_x_index_two_segments: pd.DataFrame, fill_strategy: str):
    """Check that inverse_transform doesn't change anything in forecast."""
    df, rng = df_with_missing_range_x_index_two_segments
    ts = TSDataset(df, freq=pd.infer_freq(df.index))
    imputer = TimeSeriesImputerTransform(strategy=fill_strategy)
    model = NaiveModel()
    ts.fit_transform(transforms=[imputer])
    model.fit(ts)
    ts_test = ts.make_future(3)
    assert np.all(ts_test[:, :, "target"].isna())
    ts_forecast = model.forecast(ts_test)
    for segment in ts.segments:
        true_value = ts[:, segment, "target"].values[-1]
        assert np.all(ts_forecast[:, segment, "target"] == true_value)
