from typing import Tuple

import numpy as np
import pandas as pd
import pytest

from etna.datasets import generate_ar_df
from etna.datasets.tsdataset import TSDataset
from etna.transforms import DateFlagsTransform


@pytest.fixture()
def tsdf_with_exog() -> TSDataset:
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

    timestamp = pd.date_range("2021-01-01", "2021-02-11")
    df_1 = pd.DataFrame({"timestamp": timestamp, "regressor_1": 1, "regressor_2": 2, "segment": "1"})
    df_2 = pd.DataFrame({"timestamp": timestamp[5:], "regressor_1": 3, "regressor_2": 4, "segment": "2"})
    df_exog = pd.concat([df_1, df_2], ignore_index=True)
    df_exog = TSDataset.to_dataset(df_exog)

    return df, df_exog


def test_same_ending_error_raise():
    timestamp = pd.date_range("2021-01-01", "2021-02-01")
    df1 = pd.DataFrame({"timestamp": timestamp, "target": 11, "segment": "1"})
    df2 = pd.DataFrame({"timestamp": timestamp[:-5], "target": 12, "segment": "2"})
    df = pd.concat([df1, df2], ignore_index=True)
    df = TSDataset.to_dataset(df)
    ts = TSDataset(df=df, freq="D")

    with pytest.raises(ValueError):
        ts.fit_transform([])


def test_same_ending_error_pass():
    timestamp = pd.date_range("2021-01-01", "2021-02-01")
    df1 = pd.DataFrame({"timestamp": timestamp, "target": 11, "segment": "1"})
    df2 = pd.DataFrame({"timestamp": timestamp, "target": 12, "segment": "2"})
    df = pd.concat([df1, df2], ignore_index=True)
    df = TSDataset.to_dataset(df)
    ts = TSDataset(df=df, freq="D")
    ts.fit_transform([])


def test_categorical_after_call_to_pandas():
    classic_df = generate_ar_df(periods=30, start_time="2021-06-01", n_segments=2)
    classic_df["categorical_column"] = [0] * 30 + [1] * 30
    classic_df["categorical_column"] = classic_df["categorical_column"].astype("category")
    df = TSDataset.to_dataset(classic_df[["timestamp", "segment", "target"]])
    exog = TSDataset.to_dataset(classic_df[["timestamp", "segment", "categorical_column"]])
    ts = TSDataset(df, "1d", exog)
    flatten_df = ts.to_pandas(flatten=True)
    assert flatten_df["categorical_column"].dtype == "category"


@pytest.mark.parametrize(
    "train_start,train_end,test_start,test_end",
    (("2021-02-03", "2021-06-20", "2021-06-21", "2021-07-01"), (None, "2021-06-20", "2021-06-21", "2021-07-01")),
)
def test_train_test_split(train_start, train_end, test_start, test_end, tsdf_with_exog):
    train, test = tsdf_with_exog.train_test_split(
        train_start=train_start, train_end=train_end, test_start=test_start, test_end=test_end
    )
    assert isinstance(train, TSDataset)
    assert isinstance(test, TSDataset)
    assert (train.df == tsdf_with_exog.df[train_start:train_end]).all().all()
    assert (train.df_exog == tsdf_with_exog.df_exog).all().all()
    assert (test.df == tsdf_with_exog.df[test_start:test_end]).all().all()
    assert (test.df_exog == tsdf_with_exog.df_exog).all().all()


def test_dataset_datetime_convertion():
    classic_df = generate_ar_df(periods=30, start_time="2021-06-01", n_segments=2)
    classic_df["timestamp"] = classic_df["timestamp"].astype(str)
    df = TSDataset.to_dataset(classic_df[["timestamp", "segment", "target"]])
    # todo: deal with pandas datetime format
    assert df.index.dtype == "datetime64[ns]"


def test_dataset_datetime_convertion_during_init():
    classic_df = generate_ar_df(periods=30, start_time="2021-06-01", n_segments=2)
    classic_df["categorical_column"] = [0] * 30 + [1] * 30
    classic_df["categorical_column"] = classic_df["categorical_column"].astype("category")
    df = TSDataset.to_dataset(classic_df[["timestamp", "segment", "target"]])
    exog = TSDataset.to_dataset(classic_df[["timestamp", "segment", "categorical_column"]])
    df.index = df.index.astype(str)
    exog.index = df.index.astype(str)
    ts = TSDataset(df, "1d", exog)
    assert ts.df.index.dtype == "datetime64[ns]"


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
        TSDataset._check_exog(df=df, df_exog=dfexog)


def test_dataset_check_exog_pass(df_and_regressors):
    df, df_exog = df_and_regressors
    _ = TSDataset._check_exog(df=df, df_exog=df_exog)


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
    )
    initial_regressors = set(ts.regressors)
    ts.fit_transform(transforms=[date_flags_transform])
    final_regressors = set(ts.regressors)
    assert initial_regressors.issubset(final_regressors)
    assert final_regressors.difference(initial_regressors) == {"regressor_day_number_in_week", "regressor_is_weekend"}
