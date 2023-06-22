from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from etna.datasets import TSDataset
from etna.models import NaiveModel
from etna.transforms.missing_values import TimeSeriesImputerTransform
from tests.test_transforms.utils import assert_sampling_is_valid
from tests.test_transforms.utils import assert_transformation_equals_loaded_original
from tests.utils import select_segments_subset


@pytest.fixture
def ts_nans_beginning(example_reg_tsds):
    """Example dataset with NaNs at the beginning."""
    ts = deepcopy(example_reg_tsds)

    # nans at the beginning (shouldn't be filled)
    ts.loc[ts.index[:5], pd.IndexSlice["segment_1", "target"]] = np.NaN

    # nans in the middle (should be filled)
    ts.loc[ts.index[8], pd.IndexSlice["segment_1", "target"]] = np.NaN
    ts.loc[ts.index[10], pd.IndexSlice["segment_2", "target"]] = np.NaN
    ts.loc[ts.index[40], pd.IndexSlice["segment_2", "target"]] = np.NaN
    return ts


def test_wrong_init():
    with pytest.raises(NotImplementedError, match="wrong_strategy is not a valid ImputerMode"):
        _ = TimeSeriesImputerTransform(strategy="wrong_strategy")


@pytest.mark.parametrize("fill_strategy", ["mean", "constant", "running_mean", "forward_fill", "seasonal"])
def test_transform_not_fitted(fill_strategy, ts_all_date_present_two_segments):
    transform = TimeSeriesImputerTransform(strategy=fill_strategy)
    with pytest.raises(ValueError, match="Transform is not fitted"):
        _ = transform.transform(ts_all_date_present_two_segments)


@pytest.mark.parametrize("fill_strategy", ["mean", "constant", "running_mean", "forward_fill", "seasonal"])
def test_inverse_transform_not_fitted(fill_strategy, ts_all_date_present_two_segments):
    transform = TimeSeriesImputerTransform(strategy=fill_strategy)
    with pytest.raises(ValueError, match="Transform is not fitted"):
        _ = transform.inverse_transform(ts_all_date_present_two_segments)


@pytest.mark.smoke
@pytest.mark.parametrize("fill_strategy", ["mean", "constant", "running_mean", "forward_fill", "seasonal"])
def test_all_dates_present_impute(ts_all_date_present_two_segments, fill_strategy: str):
    """Check that imputer does nothing with series without nans."""
    imputer = TimeSeriesImputerTransform(strategy=fill_strategy)
    result = imputer.fit_transform(ts_all_date_present_two_segments).to_pandas()
    for segment in result.columns.get_level_values("segment"):
        np.testing.assert_array_equal(
            ts_all_date_present_two_segments.to_pandas()[segment]["target"], result[segment]["target"]
        )


@pytest.mark.parametrize("fill_strategy", ["mean", "running_mean", "forward_fill", "seasonal"])
def test_all_missing_impute_fail(ts_all_missing_two_segments: TSDataset, fill_strategy: str):
    """Check that imputer can't fill nans if all values are nans."""
    imputer = TimeSeriesImputerTransform(strategy=fill_strategy)
    with pytest.raises(ValueError, match="Series hasn't non NaN values which means it is empty and can't be filled"):
        imputer.fit_transform(ts_all_missing_two_segments)


@pytest.mark.parametrize("constant_value", (0, 42))
def test_one_missing_value_constant(ts_with_missing_value_x_index, constant_value: float):
    """Check that imputer with constant-strategy works correctly in case of one missing value in data."""
    ts, segment, idx = ts_with_missing_value_x_index
    imputer = TimeSeriesImputerTransform(
        in_column="target",
        strategy="constant",
        constant_value=constant_value,
    )
    result = imputer.fit_transform(ts).to_pandas().loc[:, pd.IndexSlice[segment, "target"]]
    assert result.loc[idx] == constant_value
    assert not result.isna().any()


@pytest.mark.parametrize("constant_value", (0, 42))
def test_range_missing_constant(ts_with_missing_range_x_index, constant_value: float):
    """Check that imputer with constant-strategy works correctly in case of range of missing values in data."""
    ts, segment, rng = ts_with_missing_range_x_index
    imputer = TimeSeriesImputerTransform(
        in_column="target",
        strategy="constant",
        constant_value=constant_value,
    )
    result = imputer.fit_transform(ts).to_pandas().loc[:, pd.IndexSlice[segment, "target"]]
    expected_series = pd.Series(index=rng, data=[constant_value for _ in rng], name="target")
    np.testing.assert_array_almost_equal(result.loc[rng].reset_index(drop=True), expected_series)
    assert not result.isna().any()


def test_one_missing_value_mean(ts_with_missing_value_x_index):
    """Check that imputer with mean-strategy works correctly in case of one missing value in data."""
    ts, segment, idx = ts_with_missing_value_x_index
    imputer = TimeSeriesImputerTransform(in_column="target", strategy="mean")
    expected_value = ts.df.loc[:, pd.IndexSlice[segment, "target"]].mean()
    result = imputer.fit_transform(ts).to_pandas().loc[:, pd.IndexSlice[segment, "target"]]
    assert result.loc[idx] == expected_value
    assert not result.isna().any()


def test_range_missing_mean(ts_with_missing_range_x_index):
    """Check that imputer with mean-strategy works correctly in case of range of missing values in data."""
    ts, segment, rng = ts_with_missing_range_x_index
    imputer = TimeSeriesImputerTransform(in_column="target", strategy="mean")
    result = imputer.fit_transform(ts).to_pandas().loc[:, pd.IndexSlice[segment, "target"]]
    expected_value = ts.df.loc[:, pd.IndexSlice[segment, "target"]].mean()
    expected_series = pd.Series(index=rng, data=[expected_value for _ in rng], name="target")
    np.testing.assert_array_almost_equal(result.loc[rng].reset_index(drop=True), expected_series)
    assert not result.isna().any()


def test_one_missing_value_forward_fill(ts_with_missing_value_x_index):
    """Check that imputer with forward-fill-strategy works correctly in case of one missing value in data."""
    ts, segment, idx = ts_with_missing_value_x_index
    imputer = TimeSeriesImputerTransform(in_column="target", strategy="forward_fill")
    result = imputer.fit_transform(ts).to_pandas().loc[:, pd.IndexSlice[segment, "target"]]

    timestamps = np.array(sorted(ts.index))
    timestamp_idx = np.where(timestamps == idx)[0][0]
    expected_value = ts.df.loc[timestamps[timestamp_idx - 1], pd.IndexSlice[segment, "target"]]
    assert result.loc[idx] == expected_value
    assert not result.isna().any()


def test_range_missing_forward_fill(ts_with_missing_range_x_index):
    """Check that imputer with forward-fill-strategy works correctly in case of range of missing values in data."""
    ts, segment, rng = ts_with_missing_range_x_index
    imputer = TimeSeriesImputerTransform(in_column="target", strategy="forward_fill")
    result = imputer.fit_transform(ts).to_pandas().loc[:, pd.IndexSlice[segment, "target"]]

    timestamps = np.array(sorted(ts.index))
    rng = [pd.Timestamp(x) for x in rng]
    timestamp_idx = min(np.where([x in rng for x in timestamps])[0])
    expected_value = ts.df.loc[timestamps[timestamp_idx - 1], pd.IndexSlice[segment, "target"]]
    expected_series = pd.Series(index=rng, data=[expected_value for _ in rng], name="target")
    np.testing.assert_array_almost_equal(result.loc[rng], expected_series)
    assert not result.isna().any()


@pytest.mark.parametrize("window", [1, -1, 2])
def test_one_missing_value_running_mean(ts_with_missing_value_x_index, window: int):
    """Check that imputer with running-mean-strategy works correctly in case of one missing value in data."""
    ts, segment, idx = ts_with_missing_value_x_index
    timestamps = np.array(sorted(ts.index))
    timestamp_idx = np.where(timestamps == idx)[0][0]
    imputer = TimeSeriesImputerTransform(in_column="target", strategy="running_mean", window=window)
    if window == -1:
        expected_value = ts.df.loc[: timestamps[timestamp_idx - 1], pd.IndexSlice[segment, "target"]].mean()
    else:
        expected_value = ts.df.loc[
            timestamps[timestamp_idx - window] : timestamps[timestamp_idx - 1], pd.IndexSlice[segment, "target"]
        ].mean()
    result = imputer.fit_transform(ts).to_pandas().loc[:, pd.IndexSlice[segment, "target"]]
    assert result.loc[idx] == expected_value
    assert not result.isna().any()


@pytest.mark.parametrize("window", [1, -1, 2])
def test_range_missing_running_mean(ts_with_missing_range_x_index, window: int):
    """Check that imputer with running-mean-strategy works correctly in case of range of missing values in data."""
    ts, segment, rng = ts_with_missing_range_x_index
    timestamps = np.array(sorted(ts.index))
    timestamp_idxs = np.where([x in rng for x in timestamps])[0]
    imputer = TimeSeriesImputerTransform(in_column="target", strategy="running_mean", window=window)
    result = imputer.fit_transform(ts).to_pandas().loc[:, pd.IndexSlice[segment, "target"]]

    assert not result.isna().any()
    for idx in timestamp_idxs:
        if window == -1:
            expected_value = result.loc[: timestamps[idx - 1]].mean()
        else:
            expected_value = result.loc[timestamps[idx - window] : timestamps[idx - 1]].mean()
        assert result.loc[timestamps[idx]] == expected_value


@pytest.fixture
def sample_ts():
    timestamp = pd.date_range(start="2020-01-01", end="2020-01-11", freq="D")
    df1 = pd.DataFrame()
    df1["timestamp"] = timestamp
    df1["segment"] = "segment_1"
    df1["target"] = np.arange(-1, 10)

    df2 = pd.DataFrame()
    df2["timestamp"] = timestamp
    df2["segment"] = "segment_2"
    df2["target"] = np.arange(0, 110, 10)

    df = pd.concat([df1, df2], ignore_index=True)
    ts = TSDataset(df=TSDataset.to_dataset(df), freq="D")
    return ts


@pytest.fixture
def ts_to_fill(sample_ts):
    """TSDataset with nans to fill with imputer."""
    ts = deepcopy(sample_ts)
    ts.df.loc[["2020-01-01", "2020-01-03", "2020-01-08", "2020-01-09"], pd.IndexSlice[:, "target"]] = np.NaN
    return ts


@pytest.mark.parametrize(
    "window, seasonality, expected",
    [
        (
            1,
            3,
            np.array(
                [[np.NaN, 0, np.NaN, 2, 3, 4, 5, 3, 4, 8, 9], [np.NaN, 10, np.NaN, 30, 40, 50, 60, 40, 50, 90, 100]]
            ).T,
        ),
        (
            3,
            1,
            np.array(
                [[np.NaN, 0, 0, 2, 3, 4, 5, 4, 13 / 3, 8, 9], [np.NaN, 10, 10, 30, 40, 50, 60, 50, 160 / 3, 90, 100]]
            ).T,
        ),
        (
            3,
            3,
            np.array(
                [[np.NaN, 0, np.NaN, 2, 3, 4, 5, 3 / 2, 4, 8, 9], [np.NaN, 10, np.NaN, 30, 40, 50, 60, 25, 50, 90, 100]]
            ).T,
        ),
        (
            -1,
            3,
            np.array(
                [[np.NaN, 0, np.NaN, 2, 3, 4, 5, 3 / 2, 4, 8, 9], [np.NaN, 10, np.NaN, 30, 40, 50, 60, 25, 50, 90, 100]]
            ).T,
        ),
    ],
)
def test_missing_values_seasonal(ts_to_fill, window: int, seasonality: int, expected: np.ndarray):
    ts = deepcopy(ts_to_fill)
    imputer = TimeSeriesImputerTransform(
        in_column="target", strategy="seasonal", window=window, seasonality=seasonality, default_value=None
    )
    imputer.fit_transform(ts)
    result = ts.df.loc[pd.IndexSlice[:], pd.IndexSlice[:, "target"]].values

    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    "window, seasonality, default_value, expected",
    [
        (
            1,
            3,
            100,
            np.array([[np.NaN, 0, 100, 2, 3, 4, 5, 3, 4, 8, 9], [np.NaN, 10, 100, 30, 40, 50, 60, 40, 50, 90, 100]]).T,
        ),
    ],
)
def test_default_value(ts_to_fill, window: int, seasonality: int, default_value: float, expected: np.ndarray):
    ts = deepcopy(ts_to_fill)
    imputer = TimeSeriesImputerTransform(
        in_column="target", strategy="seasonal", window=window, seasonality=seasonality, default_value=default_value
    )
    imputer.fit_transform(ts)
    result = ts.df.loc[pd.IndexSlice[:], pd.IndexSlice[:, "target"]].values

    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("fill_strategy", ["mean", "constant", "running_mean", "forward_fill", "seasonal"])
def test_inverse_transform(ts_with_missing_range_x_index_two_segments: TSDataset, fill_strategy: str):
    """Check that transform + inverse_transform don't change original df for two segments."""
    ts, rng = ts_with_missing_range_x_index_two_segments
    df = ts.to_pandas()
    imputer = TimeSeriesImputerTransform(strategy=fill_strategy)
    imputer.fit_transform(ts)
    inverse_transform_result = imputer.inverse_transform(ts).to_pandas()
    np.testing.assert_array_equal(df, inverse_transform_result)


@pytest.mark.parametrize("fill_strategy", ["mean", "constant", "running_mean", "forward_fill", "seasonal"])
def test_inverse_transform_in_forecast(ts_with_missing_range_x_index_two_segments: pd.DataFrame, fill_strategy: str):
    """Check that inverse_transform doesn't change anything in forecast."""
    ts, rng = ts_with_missing_range_x_index_two_segments
    imputer = TimeSeriesImputerTransform(strategy=fill_strategy)
    model = NaiveModel()
    ts.fit_transform(transforms=[imputer])
    model.fit(ts)
    ts_test = ts.make_future(future_steps=3, transforms=[imputer], tail_steps=model.context_size)
    assert np.all(ts_test[ts_test.index[-3] :, :, "target"].isna())
    ts_forecast = model.forecast(ts_test, prediction_size=3)
    ts_forecast.inverse_transform([imputer])
    for segment in ts.segments:
        true_value = ts[:, segment, "target"].values[-1]
        assert np.all(ts_forecast[:, segment, "target"] == true_value)


@pytest.mark.parametrize("fill_strategy", ["mean", "constant", "running_mean", "forward_fill", "seasonal"])
def test_fit_transform_nans_at_the_beginning(fill_strategy, ts_nans_beginning):
    """Check that transform doesn't fill NaNs at the beginning."""
    imputer = TimeSeriesImputerTransform(in_column="target", strategy=fill_strategy)
    df_init = ts_nans_beginning.to_pandas()
    df_filled = imputer.fit_transform(ts_nans_beginning).to_pandas()
    for segment in ts_nans_beginning.segments:
        df_segment_init = df_init.loc[:, pd.IndexSlice[segment, "target"]]
        df_segment_filled = df_filled.loc[:, pd.IndexSlice[segment, "target"]]
        first_valid_index = df_segment_init.first_valid_index()
        assert df_segment_init[:first_valid_index].equals(df_segment_filled[:first_valid_index])
        assert not df_segment_filled[first_valid_index:].isna().any()


@pytest.mark.parametrize("fill_strategy", ["mean", "constant", "running_mean", "forward_fill", "seasonal"])
def test_fit_transform_nans_at_the_end(fill_strategy, ts_diff_endings):
    """Check that transform correctly works with NaNs at the end."""
    imputer = TimeSeriesImputerTransform(in_column="target", strategy=fill_strategy)
    imputer.fit_transform(ts_diff_endings)
    assert (ts_diff_endings[:, :, "target"].isna()).sum().sum() == 0


@pytest.mark.parametrize("constant_value", (0, 32))
def test_constant_fill_strategy(ts_with_missing_range_x_index_two_segments: TSDataset, constant_value: float):
    ts, rng = ts_with_missing_range_x_index_two_segments
    imputer = TimeSeriesImputerTransform(
        in_column="target", strategy="constant", constant_value=constant_value, default_value=constant_value - 1
    )
    df = imputer.fit_transform(ts).to_pandas()
    for segment in ["segment_1", "segment_2"]:
        np.testing.assert_array_equal(
            df.loc[pd.IndexSlice[rng], pd.IndexSlice[segment, "target"]].values, [constant_value] * 5
        )


def _check_same_segments(df_1: pd.DataFrame, df_2: pd.DataFrame):
    df_1_segments = set(df_1.columns.get_level_values("segment"))
    df_2_segments = set(df_2.columns.get_level_values("segment"))
    assert df_1_segments == df_2_segments


@pytest.mark.parametrize("fill_strategy", ["mean", "constant", "running_mean", "forward_fill", "seasonal"])
def test_transform_subset_segments(fill_strategy, ts_with_missing_range_x_index_two_segments):
    ts, rng = ts_with_missing_range_x_index_two_segments
    train_ts = ts
    test_ts = select_segments_subset(ts=ts, segments=["segment_1"])
    test_df = test_ts.to_pandas()
    transform = TimeSeriesImputerTransform(in_column="target", strategy=fill_strategy)

    transform.fit(train_ts)
    transformed_df = transform.transform(test_ts).to_pandas()

    _check_same_segments(transformed_df, test_df)


@pytest.mark.parametrize("fill_strategy", ["mean", "constant", "running_mean", "forward_fill", "seasonal"])
def test_inverse_transform_subset_segments(fill_strategy, ts_with_missing_range_x_index_two_segments):
    ts, rng = ts_with_missing_range_x_index_two_segments
    train_ts = ts
    test_ts = select_segments_subset(ts=ts, segments=["segment_1"])
    test_df = test_ts.to_pandas()
    transform = TimeSeriesImputerTransform(in_column="target", strategy=fill_strategy)

    transform.fit(train_ts)
    transformed_df = transform.inverse_transform(test_ts).to_pandas()

    _check_same_segments(transformed_df, test_df)


@pytest.mark.parametrize("fill_strategy", ["mean", "constant", "running_mean", "forward_fill", "seasonal"])
def test_transform_new_segments(fill_strategy, ts_with_missing_range_x_index_two_segments):
    ts, rng = ts_with_missing_range_x_index_two_segments
    train_ts = select_segments_subset(ts=ts, segments=["segment_1"])
    test_ts = select_segments_subset(ts=ts, segments=["segment_2"])
    transform = TimeSeriesImputerTransform(in_column="target", strategy=fill_strategy)

    transform.fit(train_ts)
    with pytest.raises(
        NotImplementedError, match="This transform can't process segments that weren't present on train data"
    ):
        _ = transform.transform(test_ts)


@pytest.mark.parametrize("fill_strategy", ["mean", "constant", "running_mean", "forward_fill", "seasonal"])
def test_inverse_transform_new_segments(fill_strategy, ts_with_missing_range_x_index_two_segments):
    ts, rng = ts_with_missing_range_x_index_two_segments
    train_ts = select_segments_subset(ts=ts, segments=["segment_1"])
    test_ts = select_segments_subset(ts=ts, segments=["segment_2"])
    transform = TimeSeriesImputerTransform(in_column="target", strategy=fill_strategy)

    transform.fit(train_ts)
    with pytest.raises(
        NotImplementedError, match="This transform can't process segments that weren't present on train data"
    ):
        _ = transform.inverse_transform(test_ts)


def test_save_load(ts_to_fill):
    transform = TimeSeriesImputerTransform()
    assert_transformation_equals_loaded_original(transform=transform, ts=ts_to_fill)


@pytest.mark.parametrize(
    "transform, expected_strategy_length",
    [(TimeSeriesImputerTransform(), 4), (TimeSeriesImputerTransform(seasonality=7), 5)],
)
def test_params_to_tune(transform, expected_strategy_length, ts_to_fill):
    ts = ts_to_fill
    grid = transform.params_to_tune()
    assert len(grid) > 0
    assert len(grid["strategy"].choices) == expected_strategy_length
    assert_sampling_is_valid(transform=transform, ts=ts)
