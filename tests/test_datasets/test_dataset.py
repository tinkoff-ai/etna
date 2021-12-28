from typing import Tuple

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from etna.datasets import generate_ar_df
from etna.datasets.tsdataset import TSDataset
from etna.transforms import DateFlagsTransform
from etna.transforms import TimeSeriesImputerTransform


@pytest.fixture()
def tsdf_with_exog(random_seed) -> TSDataset:
    df_1 = pd.DataFrame.from_dict({"timestamp": pd.date_range("2021-02-01", "2021-07-01", freq="1d")})
    df_2 = pd.DataFrame.from_dict({"timestamp": pd.date_range("2021-02-01", "2021-07-01", freq="1d")})
    df_1["segment"] = "Moscow"
    df_1["target"] = [x ** 2 + np.random.uniform(-2, 2) for x in list(range(len(df_1)))]
    df_2["segment"] = "Omsk"
    df_2["target"] = [x ** 0.5 + np.random.uniform(-2, 2) for x in list(range(len(df_2)))]
    classic_df = pd.concat([df_1, df_2], ignore_index=True)

    df = classic_df.pivot(index="timestamp", columns="segment")
    df = df.reorder_levels([1, 0], axis=1)
    df = df.sort_index(axis=1)
    df.columns.names = ["segment", "feature"]

    exog = generate_ar_df(start_time="2021-01-01", periods=600, n_segments=2)
    exog = exog.pivot(index="timestamp", columns="segment")
    exog = exog.reorder_levels([1, 0], axis=1)
    exog = exog.sort_index(axis=1)
    exog.columns.names = ["segment", "feature"]
    exog.columns = pd.MultiIndex.from_arrays([["Moscow", "Omsk"], ["exog", "exog"]])

    ts = TSDataset(df=df, df_exog=exog, freq="1D")
    return ts


@pytest.fixture()
def df_and_regressors() -> Tuple[pd.DataFrame, pd.DataFrame]:
    timestamp = pd.date_range("2021-01-01", "2021-02-01")
    df_1 = pd.DataFrame({"timestamp": timestamp, "target": 11, "segment": "1"})
    df_2 = pd.DataFrame({"timestamp": timestamp[5:], "target": 12, "segment": "2"})
    df = pd.concat([df_1, df_2], ignore_index=True)
    df = TSDataset.to_dataset(df)

    timestamp = pd.date_range("2020-12-01", "2021-02-11")
    df_1 = pd.DataFrame({"timestamp": timestamp, "regressor_1": 1, "regressor_2": 2, "segment": "1"})
    df_2 = pd.DataFrame({"timestamp": timestamp[5:], "regressor_1": 3, "regressor_2": 4, "segment": "2"})
    df_exog = pd.concat([df_1, df_2], ignore_index=True)
    df_exog = TSDataset.to_dataset(df_exog)

    return df, df_exog


@pytest.fixture()
def ts_future(example_reg_tsds):
    future = example_reg_tsds.make_future(10)
    return future


def test_check_endings_error_raise():
    """Check that _check_endings method raises exception if some segments end with nan."""
    timestamp = pd.date_range("2021-01-01", "2021-02-01")
    df1 = pd.DataFrame({"timestamp": timestamp, "target": 11, "segment": "1"})
    df2 = pd.DataFrame({"timestamp": timestamp[:-5], "target": 12, "segment": "2"})
    df = pd.concat([df1, df2], ignore_index=True)
    df = TSDataset.to_dataset(df)
    ts = TSDataset(df=df, freq="D")

    with pytest.raises(ValueError):
        ts._check_endings()


def test_check_endings_error_pass():
    """Check that _check_endings method passes if there is no nans at the end of all segments."""
    timestamp = pd.date_range("2021-01-01", "2021-02-01")
    df1 = pd.DataFrame({"timestamp": timestamp, "target": 11, "segment": "1"})
    df2 = pd.DataFrame({"timestamp": timestamp, "target": 12, "segment": "2"})
    df = pd.concat([df1, df2], ignore_index=True)
    df = TSDataset.to_dataset(df)
    ts = TSDataset(df=df, freq="D")
    ts._check_endings()


def test_categorical_after_call_to_pandas():
    classic_df = generate_ar_df(periods=30, start_time="2021-06-01", n_segments=2)
    classic_df["categorical_column"] = [0] * 30 + [1] * 30
    classic_df["categorical_column"] = classic_df["categorical_column"].astype("category")
    df = TSDataset.to_dataset(classic_df[["timestamp", "segment", "target"]])
    exog = TSDataset.to_dataset(classic_df[["timestamp", "segment", "categorical_column"]])
    ts = TSDataset(df, "D", exog)
    flatten_df = ts.to_pandas(flatten=True)
    assert flatten_df["categorical_column"].dtype == "category"


@pytest.mark.parametrize(
    "borders, true_borders",
    (
        (
            ("2021-02-01", "2021-06-20", "2021-06-21", "2021-07-01"),
            ("2021-02-01", "2021-06-20", "2021-06-21", "2021-07-01"),
        ),
        (
            ("2021-02-03", "2021-06-20", "2021-06-22", "2021-07-01"),
            ("2021-02-03", "2021-06-20", "2021-06-22", "2021-07-01"),
        ),
        (
            ("2021-02-01", "2021-06-20", "2021-06-21", "2021-06-28"),
            ("2021-02-01", "2021-06-20", "2021-06-21", "2021-06-28"),
        ),
        (
            ("2021-02-01", "2021-06-20", "2021-06-23", "2021-07-01"),
            ("2021-02-01", "2021-06-20", "2021-06-23", "2021-07-01"),
        ),
        ((None, "2021-06-20", "2021-06-23", "2021-06-28"), ("2021-02-01", "2021-06-20", "2021-06-23", "2021-06-28")),
        (("2021-02-03", "2021-06-20", "2021-06-23", None), ("2021-02-03", "2021-06-20", "2021-06-23", "2021-07-01")),
        ((None, "2021-06-20", "2021-06-23", None), ("2021-02-01", "2021-06-20", "2021-06-23", "2021-07-01")),
        ((None, "2021-06-20", None, None), ("2021-02-01", "2021-06-20", "2021-06-21", "2021-07-01")),
        ((None, None, "2021-06-21", None), ("2021-02-01", "2021-06-20", "2021-06-21", "2021-07-01")),
    ),
)
def test_train_test_split(borders, true_borders, tsdf_with_exog):
    train_start, train_end, test_start, test_end = borders
    train_start_true, train_end_true, test_start_true, test_end_true = true_borders
    train, test = tsdf_with_exog.train_test_split(
        train_start=train_start, train_end=train_end, test_start=test_start, test_end=test_end
    )
    assert isinstance(train, TSDataset)
    assert isinstance(test, TSDataset)
    assert (train.df == tsdf_with_exog.df[train_start_true:train_end_true]).all().all()
    assert (train.df_exog == tsdf_with_exog.df_exog).all().all()
    assert (test.df == tsdf_with_exog.df[test_start_true:test_end_true]).all().all()
    assert (test.df_exog == tsdf_with_exog.df_exog).all().all()


@pytest.mark.parametrize(
    "test_size, true_borders",
    (
        (11, ("2021-02-01", "2021-06-20", "2021-06-21", "2021-07-01")),
        (9, ("2021-02-01", "2021-06-22", "2021-06-23", "2021-07-01")),
        (1, ("2021-02-01", "2021-06-30", "2021-07-01", "2021-07-01")),
    ),
)
def test_train_test_split_with_test_size(test_size, true_borders, tsdf_with_exog):
    train_start_true, train_end_true, test_start_true, test_end_true = true_borders
    train, test = tsdf_with_exog.train_test_split(test_size=test_size)
    assert isinstance(train, TSDataset)
    assert isinstance(test, TSDataset)
    assert (train.df == tsdf_with_exog.df[train_start_true:train_end_true]).all().all()
    assert (train.df_exog == tsdf_with_exog.df_exog).all().all()
    assert (test.df == tsdf_with_exog.df[test_start_true:test_end_true]).all().all()
    assert (test.df_exog == tsdf_with_exog.df_exog).all().all()


@pytest.mark.parametrize(
    "test_size, borders, true_borders",
    (
        (
            10,
            ("2021-02-01", "2021-06-20", "2021-06-21", "2021-07-01"),
            ("2021-02-01", "2021-06-20", "2021-06-21", "2021-07-01"),
        ),
        (
            15,
            ("2021-02-03", "2021-06-20", "2021-06-22", "2021-07-01"),
            ("2021-02-03", "2021-06-20", "2021-06-22", "2021-07-01"),
        ),
        (11, ("2021-02-02", None, None, "2021-06-28"), ("2021-02-02", "2021-06-17", "2021-06-18", "2021-06-28")),
        (
            4,
            ("2021-02-03", "2021-06-20", None, "2021-07-01"),
            ("2021-02-03", "2021-06-20", "2021-06-28", "2021-07-01"),
        ),
        (
            4,
            ("2021-02-03", "2021-06-20", None, None),
            ("2021-02-03", "2021-06-20", "2021-06-21", "2021-06-24"),
        ),
    ),
)
def test_train_test_split_both(test_size, borders, true_borders, tsdf_with_exog):
    train_start, train_end, test_start, test_end = borders
    train_start_true, train_end_true, test_start_true, test_end_true = true_borders
    train, test = tsdf_with_exog.train_test_split(
        train_start=train_start, train_end=train_end, test_start=test_start, test_end=test_end, test_size=test_size
    )
    assert isinstance(train, TSDataset)
    assert isinstance(test, TSDataset)
    assert (train.df == tsdf_with_exog.df[train_start_true:train_end_true]).all().all()
    assert (train.df_exog == tsdf_with_exog.df_exog).all().all()
    assert (test.df == tsdf_with_exog.df[test_start_true:test_end_true]).all().all()
    assert (test.df_exog == tsdf_with_exog.df_exog).all().all()


@pytest.mark.parametrize(
    "borders, match",
    (
        (("2021-01-01", "2021-06-20", "2021-06-21", "2021-07-01"), "Min timestamp in df is"),
        (("2021-02-01", "2021-06-20", "2021-06-21", "2021-08-01"), "Max timestamp in df is"),
    ),
)
def test_train_test_split_warning(borders, match, tsdf_with_exog):
    train_start, train_end, test_start, test_end = borders
    with pytest.warns(UserWarning, match=match):
        tsdf_with_exog.train_test_split(
            train_start=train_start, train_end=train_end, test_start=test_start, test_end=test_end
        )


@pytest.mark.parametrize(
    "test_size, borders, match",
    (
        (
            10,
            ("2021-02-01", None, "2021-06-21", "2021-07-01"),
            "test_size, test_start and test_end cannot be applied at the same time. test_size will be ignored",
        ),
    ),
)
def test_train_test_split_warning2(test_size, borders, match, tsdf_with_exog):
    train_start, train_end, test_start, test_end = borders
    with pytest.warns(UserWarning, match=match):
        tsdf_with_exog.train_test_split(
            train_start=train_start, train_end=train_end, test_start=test_start, test_end=test_end, test_size=test_size
        )


@pytest.mark.parametrize(
    "test_size, borders, match",
    (
        (
            None,
            ("2021-02-03", None, None, "2021-07-01"),
            "At least one of train_end, test_start or test_size should be defined",
        ),
        (
            17,
            ("2021-02-01", "2021-06-20", None, "2021-07-01"),
            "The beginning of the test goes before the end of the train",
        ),
        (
            17,
            ("2021-02-01", "2021-06-20", "2021-06-26", None),
            "test_size is 17, but only 6 available with your test_start",
        ),
    ),
)
def test_train_test_split_failed(test_size, borders, match, tsdf_with_exog):
    train_start, train_end, test_start, test_end = borders
    with pytest.raises(ValueError, match=match):
        tsdf_with_exog.train_test_split(
            train_start=train_start, train_end=train_end, test_start=test_start, test_end=test_end, test_size=test_size
        )


def test_dataset_datetime_conversion():
    classic_df = generate_ar_df(periods=30, start_time="2021-06-01", n_segments=2)
    classic_df["timestamp"] = classic_df["timestamp"].astype(str)
    df = TSDataset.to_dataset(classic_df[["timestamp", "segment", "target"]])
    # todo: deal with pandas datetime format
    assert df.index.dtype == "datetime64[ns]"


def test_dataset_datetime_conversion_during_init():
    classic_df = generate_ar_df(periods=30, start_time="2021-06-01", n_segments=2)
    classic_df["categorical_column"] = [0] * 30 + [1] * 30
    classic_df["categorical_column"] = classic_df["categorical_column"].astype("category")
    df = TSDataset.to_dataset(classic_df[["timestamp", "segment", "target"]])
    exog = TSDataset.to_dataset(classic_df[["timestamp", "segment", "categorical_column"]])
    df.index = df.index.astype(str)
    exog.index = df.index.astype(str)
    ts = TSDataset(df, "D", exog)
    assert ts.df.index.dtype == "datetime64[ns]"


def test_make_future_raise_error_on_diff_endings(ts_diff_endings):
    with pytest.raises(ValueError, match="All segments should end at the same timestamp"):
        ts_diff_endings.make_future(10)


def test_make_future_with_imputer(ts_diff_endings, ts_future):
    imputer = TimeSeriesImputerTransform(in_column="target")
    ts_diff_endings.fit_transform([imputer])
    future = ts_diff_endings.make_future(10)
    assert_frame_equal(future.df, ts_future.df)


def test_make_future():
    timestamp = pd.date_range("2020-01-01", periods=100, freq="D")
    df1 = pd.DataFrame({"timestamp": timestamp, "target": 1, "segment": "segment_1"})
    df2 = pd.DataFrame({"timestamp": timestamp, "target": 2, "segment": "segment_2"})
    df = pd.concat([df1, df2], ignore_index=False)
    ts = TSDataset(TSDataset.to_dataset(df), freq="D")
    ts_future = ts.make_future(10)
    assert np.all(ts_future.index == pd.date_range(ts.index.max() + pd.Timedelta("1D"), periods=10, freq="D"))
    assert set(ts_future.columns.get_level_values("feature")) == {"target"}


def test_make_future_small_horizon():
    timestamp = np.arange(np.datetime64("2021-01-01"), np.datetime64("2021-02-01"))
    target1 = [np.sin(i) for i in range(len(timestamp))]
    target2 = [np.cos(i) for i in range(len(timestamp))]
    df1 = pd.DataFrame({"timestamp": timestamp, "target": target1, "segment": "1"})
    df2 = pd.DataFrame({"timestamp": timestamp, "target": target2, "segment": "2"})
    df = pd.concat([df1, df2], ignore_index=True)
    df = TSDataset.to_dataset(df)
    ts = TSDataset(df, freq="D")
    train = TSDataset(ts[: ts.index[10], :, :], freq="D")
    with pytest.warns(UserWarning, match="TSDataset freq can't be inferred"):
        assert len(train.make_future(1).df) == 1


def test_make_future_with_exog():
    timestamp = pd.date_range("2020-01-01", periods=100, freq="D")
    df1 = pd.DataFrame({"timestamp": timestamp, "target": 1, "segment": "segment_1"})
    df2 = pd.DataFrame({"timestamp": timestamp, "target": 2, "segment": "segment_2"})
    df = pd.concat([df1, df2], ignore_index=False)
    exog = df.copy()
    exog.columns = ["timestamp", "exog", "segment"]
    ts = TSDataset(df=TSDataset.to_dataset(df), df_exog=TSDataset.to_dataset(exog), freq="D")
    ts_future = ts.make_future(10)
    assert np.all(ts_future.index == pd.date_range(ts.index.max() + pd.Timedelta("1D"), periods=10, freq="D"))
    assert set(ts_future.columns.get_level_values("feature")) == {"target", "exog"}


def test_make_future_with_regressors(df_and_regressors):
    df, df_exog = df_and_regressors
    ts = TSDataset(df=df, df_exog=df_exog, freq="D")
    ts_future = ts.make_future(10)
    assert np.all(ts_future.index == pd.date_range(ts.index.max() + pd.Timedelta("1D"), periods=10, freq="D"))
    assert set(ts_future.columns.get_level_values("feature")) == {"target", "regressor_1", "regressor_2"}


@pytest.mark.parametrize("exog_starts_later,exog_ends_earlier", ((True, False), (False, True), (True, True)))
def test_dataset_check_exog_raise_error(exog_starts_later: bool, exog_ends_earlier: bool):
    start_time = "2021-01-10" if exog_starts_later else "2021-01-01"
    end_time = "2021-01-20" if exog_ends_earlier else "2021-02-01"
    timestamp = pd.date_range("2021-01-01", "2021-02-01")
    df1 = pd.DataFrame({"timestamp": timestamp, "target": 11, "segment": "1"})
    df2 = pd.DataFrame({"timestamp": timestamp[5:], "target": 12, "segment": "2"})
    df = pd.concat([df1, df2], ignore_index=True)
    df = TSDataset.to_dataset(df)

    timestamp = pd.date_range(start_time, end_time)
    df1 = pd.DataFrame({"timestamp": timestamp, "regressor_aaa": 1, "segment": "1"})
    df2 = pd.DataFrame({"timestamp": timestamp[5:], "regressor_aaa": 2, "segment": "2"})
    dfexog = pd.concat([df1, df2], ignore_index=True)
    dfexog = TSDataset.to_dataset(dfexog)

    with pytest.raises(ValueError):
        TSDataset._check_regressors(df=df, df_exog=dfexog)


def test_dataset_check_exog_pass(df_and_regressors):
    df, df_exog = df_and_regressors
    _ = TSDataset._check_regressors(df=df, df_exog=df_exog)


def test_warn_not_enough_exog(df_and_regressors):
    """Check that warning is thrown if regressors don't have enough values."""
    df, df_exog = df_and_regressors
    ts = TSDataset(df=df, df_exog=df_exog, freq="D")
    with pytest.warns(UserWarning, match="Some regressors don't have enough values"):
        ts.make_future(ts.df_exog.shape[0] + 100)


def test_getitem_only_date(tsdf_with_exog):
    df_date_only = tsdf_with_exog["2021-02-01"]
    assert df_date_only.name == pd.Timestamp("2021-02-01")
    pd.testing.assert_series_equal(tsdf_with_exog.df.loc["2021-02-01"], df_date_only)


def test_getitem_slice_date(tsdf_with_exog):
    df_slice = tsdf_with_exog["2021-02-01":"2021-02-03"]
    expected_index = pd.DatetimeIndex(pd.date_range("2021-02-01", "2021-02-03"), name="timestamp")
    pd.testing.assert_index_equal(df_slice.index, expected_index)
    pd.testing.assert_frame_equal(tsdf_with_exog.df.loc["2021-02-01":"2021-02-03"], df_slice)


def test_getitem_second_ellipsis(tsdf_with_exog):
    df_slice = tsdf_with_exog["2021-02-01":"2021-02-03", ...]
    expected_index = pd.DatetimeIndex(pd.date_range("2021-02-01", "2021-02-03"), name="timestamp")
    pd.testing.assert_index_equal(df_slice.index, expected_index)
    pd.testing.assert_frame_equal(tsdf_with_exog.df.loc["2021-02-01":"2021-02-03"], df_slice)


def test_getitem_first_ellipsis(tsdf_with_exog):
    df_slice = tsdf_with_exog[..., "target"]
    df_expected = tsdf_with_exog.df.loc[:, [["Moscow", "target"], ["Omsk", "target"]]]
    pd.testing.assert_frame_equal(df_expected, df_slice)


def test_getitem_all_indexes(tsdf_with_exog):
    df_slice = tsdf_with_exog[:, :, :]
    df_expected = tsdf_with_exog.df
    pd.testing.assert_frame_equal(df_expected, df_slice)


def test_finding_regressors(df_and_regressors):
    """Check that ts.regressors property works correctly."""
    df, df_exog = df_and_regressors
    ts = TSDataset(df=df, df_exog=df_exog, freq="D")
    assert sorted(ts.regressors) == ["regressor_1", "regressor_2"]


def test_head_default(tsdf_with_exog):
    assert np.all(tsdf_with_exog.head() == tsdf_with_exog.df.head())


def test_tail_default(tsdf_with_exog):
    np.all(tsdf_with_exog.tail() == tsdf_with_exog.df.tail())


def test_updating_regressors_fit_transform(df_and_regressors):
    """Check that ts.regressors is updated after making ts.fit_transform()."""
    df, df_exog = df_and_regressors
    ts = TSDataset(df=df, df_exog=df_exog, freq="D")
    date_flags_transform = DateFlagsTransform(
        day_number_in_week=True,
        day_number_in_month=False,
        week_number_in_month=False,
        week_number_in_year=False,
        month_number_in_year=False,
        year_number=False,
        is_weekend=True,
        out_column="regressor_dateflag",
    )
    initial_regressors = set(ts.regressors)
    ts.fit_transform(transforms=[date_flags_transform])
    final_regressors = set(ts.regressors)
    expected_columns = {"regressor_dateflag_day_number_in_week", "regressor_dateflag_is_weekend"}
    assert initial_regressors.issubset(final_regressors)
    assert final_regressors.difference(initial_regressors) == expected_columns


def test_right_format_sorting():
    """Need to check if to_dataset method does not mess up with data and column names,
    sorting it with no respect to each other
    """
    df = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=100)})
    df["segment"] = "segment_1"
    # need names and values in inverse fashion
    df["reg_2"] = 1
    df["reg_1"] = 2
    tsd = TSDataset(TSDataset.to_dataset(df), freq="D")
    inv_df = tsd.to_pandas(flatten=True)
    pd.testing.assert_series_equal(df["reg_1"], inv_df["reg_1"])
    pd.testing.assert_series_equal(df["reg_2"], inv_df["reg_2"])


def test_to_flatten(example_df):
    """Check that TSDataset.to_flatten works correctly."""
    sorted_columns = sorted(example_df.columns)
    expected_df = example_df[sorted_columns]
    obtained_df = TSDataset.to_flatten(TSDataset.to_dataset(example_df))
    assert sorted_columns == sorted(obtained_df.columns)
    assert (expected_df.values == obtained_df[sorted_columns].values).all()


def test_transform_raise_warning_on_diff_endings(ts_diff_endings):
    with pytest.warns(Warning, match="Segments contains NaNs in the last timestamps."):
        ts_diff_endings.transform([])


def test_fit_transform_raise_warning_on_diff_endings(ts_diff_endings):
    with pytest.warns(Warning, match="Segments contains NaNs in the last timestamps."):
        ts_diff_endings.fit_transform([])


def test_gather_common_data(df_and_regressors):
    """Check that TSDataset._gather_common_data correctly finds common data for info/describe methods."""
    df, df_exog = df_and_regressors
    ts = TSDataset(df=df, df_exog=df_exog, freq="D")
    common_data = ts._gather_common_data()
    assert common_data["num_segments"] == 2
    assert common_data["num_exogs"] == 2
    assert common_data["num_regressors"] == 2
    assert common_data["freq"] == "D"


def test_gather_segments_data(df_and_regressors):
    """Check that TSDataset._gather_segments_data correctly finds segment data for info/describe methods."""
    df, df_exog = df_and_regressors
    # add NaN in the middle
    df.iloc[-5, 0] = np.NaN
    # add NaNs at the end
    df.iloc[-3:, 1] = np.NaN
    ts = TSDataset(df=df, df_exog=df_exog, freq="D")
    segments = ts.segments
    segments_dict = ts._gather_segments_data(segments)
    segment_df = pd.DataFrame(segments_dict, index=segments)

    assert np.all(segment_df.index == ts.segments)
    assert segment_df.loc["1", "start_timestamp"] == pd.Timestamp("2021-01-01")
    assert segment_df.loc["2", "start_timestamp"] == pd.Timestamp("2021-01-06")
    assert segment_df.loc["1", "end_timestamp"] == pd.Timestamp("2021-02-01")
    assert segment_df.loc["2", "end_timestamp"] == pd.Timestamp("2021-01-29")
    assert segment_df.loc["1", "length"] == 32
    assert segment_df.loc["2", "length"] == 24
    assert segment_df.loc["1", "num_missing"] == 1
    assert segment_df.loc["2", "num_missing"] == 0


def test_describe(df_and_regressors):
    """Check that TSDataset.describe works correctly."""
    df, df_exog = df_and_regressors
    # add NaN in the middle
    df.iloc[-5, 0] = np.NaN
    # add NaNs at the end
    df.iloc[-3:, 1] = np.NaN
    ts = TSDataset(df=df, df_exog=df_exog, freq="D")
    description = ts.describe()

    assert np.all(description.index == ts.segments)
    assert description.loc["1", "start_timestamp"] == pd.Timestamp("2021-01-01")
    assert description.loc["2", "start_timestamp"] == pd.Timestamp("2021-01-06")
    assert description.loc["1", "end_timestamp"] == pd.Timestamp("2021-02-01")
    assert description.loc["2", "end_timestamp"] == pd.Timestamp("2021-01-29")
    assert description.loc["1", "length"] == 32
    assert description.loc["2", "length"] == 24
    assert description.loc["1", "num_missing"] == 1
    assert description.loc["2", "num_missing"] == 0
    assert np.all(description["num_segments"] == 2)
    assert np.all(description["num_exogs"] == 2)
    assert np.all(description["num_regressors"] == 2)
    assert np.all(description["freq"] == "D")
