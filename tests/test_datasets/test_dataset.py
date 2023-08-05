from contextlib import suppress
from typing import List
from typing import Tuple

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from etna.datasets import generate_ar_df
from etna.datasets.tsdataset import TSDataset
from etna.transforms import AddConstTransform
from etna.transforms import DifferencingTransform
from etna.transforms import TimeSeriesImputerTransform


@pytest.fixture()
def tsdf_with_exog(random_seed) -> TSDataset:
    df_1 = pd.DataFrame.from_dict({"timestamp": pd.date_range("2021-02-01", "2021-07-01", freq="1d")})
    df_2 = pd.DataFrame.from_dict({"timestamp": pd.date_range("2021-02-01", "2021-07-01", freq="1d")})
    df_1["segment"] = "Moscow"
    df_1["target"] = [x**2 + np.random.uniform(-2, 2) for x in list(range(len(df_1)))]
    df_2["segment"] = "Omsk"
    df_2["target"] = [x**0.5 + np.random.uniform(-2, 2) for x in list(range(len(df_2)))]
    classic_df = pd.concat([df_1, df_2], ignore_index=True)

    df = TSDataset.to_dataset(classic_df)

    classic_df_exog = generate_ar_df(start_time="2021-01-01", periods=600, n_segments=2)
    classic_df_exog["segment"] = classic_df_exog["segment"].apply(lambda x: "Moscow" if x == "segment_0" else "Omsk")
    classic_df_exog.rename(columns={"target": "exog"}, inplace=True)
    df_exog = TSDataset.to_dataset(classic_df_exog)

    ts = TSDataset(df=df, df_exog=df_exog, freq="1D")
    return ts


@pytest.fixture
def df_and_regressors() -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
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

    return df, df_exog, ["regressor_1", "regressor_2"]


@pytest.fixture
def ts_info() -> TSDataset:
    timestamp = pd.date_range("2021-01-01", "2021-02-01")
    df_1 = pd.DataFrame({"timestamp": timestamp, "target": 11, "segment": "1"})
    df_2 = pd.DataFrame({"timestamp": timestamp[5:], "target": 12, "segment": "2"})
    df_3 = pd.DataFrame({"timestamp": timestamp, "target": np.NaN, "segment": "3"})
    df = pd.concat([df_1, df_2, df_3], ignore_index=True)
    df = TSDataset.to_dataset(df)

    timestamp = pd.date_range("2020-12-01", "2021-02-11")
    df_1 = pd.DataFrame({"timestamp": timestamp, "regressor_1": 1, "regressor_2": 2, "segment": "1"})
    df_2 = pd.DataFrame({"timestamp": timestamp[5:], "regressor_1": 3, "regressor_2": 4, "segment": "2"})
    df_3 = pd.DataFrame({"timestamp": timestamp, "regressor_1": 5, "regressor_2": 6, "segment": "3"})
    df_exog = pd.concat([df_1, df_2, df_3], ignore_index=True)
    df_exog = TSDataset.to_dataset(df_exog)

    # add NaN in the middle
    df.iloc[-5, 0] = np.NaN
    # add NaNs at the end
    df.iloc[-3:, 1] = np.NaN

    ts = TSDataset(df=df, df_exog=df_exog, freq="D", known_future=["regressor_1", "regressor_2"])
    return ts


@pytest.fixture
def df_update_add_column() -> pd.DataFrame:
    timestamp = pd.date_range("2021-01-01", "2021-02-12")
    df_1 = pd.DataFrame({"timestamp": timestamp, "new_column": 100, "segment": "1"})
    df_2 = pd.DataFrame({"timestamp": timestamp, "new_column": 200, "segment": "2"})
    df = pd.concat([df_1, df_2], ignore_index=True)
    df = TSDataset.to_dataset(df)
    return df


@pytest.fixture
def df_update_update_column() -> pd.DataFrame:
    timestamp = pd.date_range("2021-01-01", "2021-02-12")
    df_1 = pd.DataFrame({"timestamp": timestamp, "target": 100, "segment": "1"})
    df_2 = pd.DataFrame({"timestamp": timestamp, "target": 200, "segment": "2"})
    df = pd.concat([df_1, df_2], ignore_index=True)
    df = TSDataset.to_dataset(df)
    return df


@pytest.fixture
def df_updated_add_column() -> pd.DataFrame:
    timestamp = pd.date_range("2021-01-01", "2021-02-01")
    df_1 = pd.DataFrame({"timestamp": timestamp, "target": 11, "new_column": 100, "segment": "1"})
    df_2 = pd.DataFrame({"timestamp": timestamp, "target": 12, "new_column": 200, "segment": "2"})
    df_2.loc[:4, "target"] = None
    df = pd.concat([df_1, df_2], ignore_index=True)
    df = TSDataset(df=TSDataset.to_dataset(df), freq="D").df
    return df


@pytest.fixture
def df_updated_update_column() -> pd.DataFrame:
    timestamp = pd.date_range("2021-01-01", "2021-02-01")
    df_1 = pd.DataFrame({"timestamp": timestamp, "target": 100, "segment": "1"})
    df_2 = pd.DataFrame({"timestamp": timestamp, "target": 200, "segment": "2"})
    df = pd.concat([df_1, df_2], ignore_index=True)
    df = TSDataset(df=TSDataset.to_dataset(df), freq="D").df
    return df


@pytest.fixture
def df_exog_updated_add_column() -> pd.DataFrame:
    timestamp = pd.date_range("2020-12-01", "2021-02-12")
    df_1 = pd.DataFrame({"timestamp": timestamp, "regressor_1": 1, "regressor_2": 2, "new_column": 100, "segment": "1"})
    df_1.iloc[-1:, df_1.columns.get_loc("regressor_1")] = None
    df_1.iloc[-1:, df_1.columns.get_loc("regressor_2")] = None
    df_1.iloc[:31, df_1.columns.get_loc("new_column")] = None
    df_2 = pd.DataFrame({"timestamp": timestamp, "regressor_1": 3, "regressor_2": 4, "new_column": 200, "segment": "2"})
    df_2.iloc[:5, df_2.columns.get_loc("regressor_1")] = None
    df_2.iloc[:5, df_2.columns.get_loc("regressor_2")] = None
    df_2.iloc[-1:, df_2.columns.get_loc("regressor_1")] = None
    df_2.iloc[-1:, df_2.columns.get_loc("regressor_2")] = None
    df_2.iloc[:31, df_2.columns.get_loc("new_column")] = None
    df_exog = pd.concat([df_1, df_2], ignore_index=True)
    df_exog = TSDataset.to_dataset(df_exog)
    df_exog = TSDataset(df=df_exog, freq="D").df
    return df_exog


@pytest.fixture
def df_and_regressors_flat() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return flat versions of df and df_exog."""
    timestamp = pd.date_range("2021-01-01", "2021-02-01")
    df_1 = pd.DataFrame({"timestamp": timestamp, "target": 11, "segment": "1"})
    df_2 = pd.DataFrame({"timestamp": timestamp[5:], "target": 12, "segment": "2"})
    df = pd.concat([df_1, df_2], ignore_index=True)

    timestamp = pd.date_range("2020-12-01", "2021-02-11")
    df_1 = pd.DataFrame(
        {"timestamp": timestamp, "regressor_1": 1, "regressor_2": "3", "regressor_3": 5, "segment": "1"}
    )
    df_2 = pd.DataFrame(
        {"timestamp": timestamp[5:], "regressor_1": 2, "regressor_2": "4", "regressor_3": 6, "segment": "2"}
    )
    df_exog = pd.concat([df_1, df_2], ignore_index=True)
    df_exog["regressor_2"] = df_exog["regressor_2"].astype("category")
    df_exog["regressor_3"] = df_exog["regressor_3"].astype("category")

    return df, df_exog


@pytest.fixture
def ts_with_categoricals():
    timestamp = pd.date_range("2021-01-01", "2021-01-05")
    df_1 = pd.DataFrame({"timestamp": timestamp, "target": 11, "segment": "1"})
    df_2 = pd.DataFrame({"timestamp": timestamp, "target": 12, "segment": "2"})
    df = pd.concat([df_1, df_2], ignore_index=True)
    df = TSDataset.to_dataset(df)

    timestamp = pd.date_range("2021-01-01", "2021-01-06")
    categorical_values = ["1", "2", "1", "2", "1", "2"]
    df_1 = pd.DataFrame(
        {"timestamp": timestamp, "regressor": categorical_values, "not_regressor": categorical_values, "segment": "1"}
    )
    df_2 = pd.DataFrame(
        {"timestamp": timestamp, "regressor": categorical_values, "not_regressor": categorical_values, "segment": "2"}
    )
    df_exog = pd.concat([df_1, df_2], ignore_index=True)
    df_exog = TSDataset.to_dataset(df_exog)

    ts = TSDataset(df=df, freq="D", df_exog=df_exog, known_future=["regressor"])
    return ts


@pytest.fixture()
def ts_future(example_reg_tsds):
    future = example_reg_tsds.make_future(10)
    return future


@pytest.fixture
def df_segments_int():
    """DataFrame with integer segments."""
    timestamp = pd.date_range("2021-01-01", "2021-02-01")
    df1 = pd.DataFrame({"timestamp": timestamp, "target": 3, "segment": 1})
    df2 = pd.DataFrame({"timestamp": timestamp, "target": 4, "segment": 2})
    df = pd.concat([df1, df2], ignore_index=True)
    return df


@pytest.fixture
def target_components_df():
    timestamp = pd.date_range("2021-01-01", "2021-01-15")
    df_1 = pd.DataFrame({"timestamp": timestamp, "target_component_a": 1, "target_component_b": 2, "segment": 1})
    df_2 = pd.DataFrame({"timestamp": timestamp, "target_component_a": 3, "target_component_b": 4, "segment": 2})
    df = pd.concat([df_1, df_2])
    df = TSDataset.to_dataset(df)
    return df


@pytest.fixture
def inverse_transformed_components_df():
    timestamp = pd.date_range("2021-01-01", "2021-01-15")
    df_1 = pd.DataFrame(
        {
            "timestamp": timestamp,
            "target_component_a": 1 * (3 + 10) / 3,
            "target_component_b": 2 * (3 + 10) / 3,
            "segment": 1,
        }
    )
    df_2 = pd.DataFrame(
        {
            "timestamp": timestamp,
            "target_component_a": 3 * (7 + 10) / 7,
            "target_component_b": 4 * (7 + 10) / 7,
            "segment": 2,
        }
    )
    df = pd.concat([df_1, df_2])
    df = TSDataset.to_dataset(df)
    df.index.freq = "D"
    return df


@pytest.fixture
def inconsistent_target_components_names_df(target_components_df):
    target_components_df = target_components_df.drop(columns=[("2", "target_component_a")])
    return target_components_df


@pytest.fixture
def inconsistent_target_components_names_duplication_df(target_components_df):
    target_components_df = pd.concat(
        (target_components_df, target_components_df.loc[pd.IndexSlice[:], pd.IndexSlice["1", :]]), axis=1
    )
    return target_components_df


@pytest.fixture
def inconsistent_target_components_values_df(target_components_df):
    target_components_df.loc[target_components_df.index[-1], pd.IndexSlice["1", "target_component_a"]] = 100
    target_components_df.loc[target_components_df.index[10], pd.IndexSlice["1", "target_component_a"]] = 100
    return target_components_df


@pytest.fixture
def ts_without_target_components():
    timestamp = pd.date_range("2021-01-01", "2021-01-15")
    df_1 = pd.DataFrame({"timestamp": timestamp, "target": 3, "segment": 1})
    df_2 = pd.DataFrame({"timestamp": timestamp, "target": 7, "segment": 2})
    df = pd.concat([df_1, df_2])
    df = TSDataset.to_dataset(df)
    ts = TSDataset(df=df, freq="D")
    return ts


@pytest.fixture
def ts_with_target_components():
    timestamp = pd.date_range("2021-01-01", "2021-01-15")
    df_1 = pd.DataFrame(
        {"timestamp": timestamp, "target": 3, "target_component_a": 1, "target_component_b": 2, "segment": 1}
    )
    df_2 = pd.DataFrame(
        {"timestamp": timestamp, "target": 7, "target_component_a": 3, "target_component_b": 4, "segment": 2}
    )
    df = pd.concat([df_1, df_2])
    df = TSDataset.to_dataset(df)
    ts = TSDataset(df=df, freq="D")
    ts._target_components_names = ["target_component_a", "target_component_b"]
    return ts


def test_check_endings_error():
    """Check that _check_endings method raises exception if some segments end with nan."""
    timestamp = pd.date_range("2021-01-01", "2021-02-01")
    df1 = pd.DataFrame({"timestamp": timestamp, "target": 11, "segment": "1"})
    df2 = pd.DataFrame({"timestamp": timestamp[:-5], "target": 12, "segment": "2"})
    df = pd.concat([df1, df2], ignore_index=True)
    df = TSDataset.to_dataset(df)
    ts = TSDataset(df=df, freq="D")

    with pytest.raises(ValueError):
        ts._check_endings()


def test_check_endings_pass():
    """Check that _check_endings method passes if there is no nans at the end of all segments."""
    timestamp = pd.date_range("2021-01-01", "2021-02-01")
    df1 = pd.DataFrame({"timestamp": timestamp, "target": 11, "segment": "1"})
    df2 = pd.DataFrame({"timestamp": timestamp, "target": 12, "segment": "2"})
    df = pd.concat([df1, df2], ignore_index=True)
    df = TSDataset.to_dataset(df)
    ts = TSDataset(df=df, freq="D")
    ts._check_endings()


def test_check_known_future_wrong_literal():
    """Check that _check_known_future raises exception if wrong literal is given."""
    with pytest.raises(ValueError, match="The only possible literal is 'all'"):
        _ = TSDataset._check_known_future("wrong-literal", None)


def test_check_known_future_error_no_df_exog():
    """Check that _check_known_future raises exception if there are no df_exog, but known_future isn't empty."""
    with pytest.raises(ValueError, match="Some features in known_future are not present in df_exog"):
        _ = TSDataset._check_known_future(["regressor_1"], None)


def test_check_known_future_error_not_matching(df_and_regressors):
    """Check that _check_known_future raises exception if df_exog doesn't contain some features in known_future."""
    _, df_exog, known_future = df_and_regressors
    known_future.append("regressor_new")
    with pytest.raises(ValueError, match="Some features in known_future are not present in df_exog"):
        _ = TSDataset._check_known_future(known_future, df_exog)


def test_check_known_future_pass_all_empty():
    """Check that _check_known_future passes if known_future and df_exog are empty."""
    regressors = TSDataset._check_known_future([], None)
    assert len(regressors) == 0


@pytest.mark.parametrize(
    "known_future, expected_columns",
    [
        ([], []),
        (["regressor_1"], ["regressor_1"]),
        (["regressor_1", "regressor_2"], ["regressor_1", "regressor_2"]),
        (["regressor_1", "regressor_1"], ["regressor_1"]),
        ("all", ["regressor_1", "regressor_2"]),
    ],
)
def test_check_known_future_pass_non_empty(df_and_regressors, known_future, expected_columns):
    _, df_exog, _ = df_and_regressors
    """Check that _check_known_future passes if df_exog is not empty."""
    regressors = TSDataset._check_known_future(known_future, df_exog)
    assert regressors == expected_columns


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


def test_train_test_split_pass_regressors_to_output(df_and_regressors):
    df, df_exog, known_future = df_and_regressors
    ts = TSDataset(df=df, df_exog=df_exog, freq="D", known_future=known_future)
    train, test = ts.train_test_split(test_size=5)
    assert train.regressors == ts.regressors
    assert test.regressors == ts.regressors


def test_train_test_split_pass_target_components_to_output(ts_with_target_components):
    train, test = ts_with_target_components.train_test_split(test_size=5)
    assert sorted(train.target_components_names) == sorted(ts_with_target_components.target_components_names)
    assert sorted(test.target_components_names) == sorted(ts_with_target_components.target_components_names)


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


def test_to_dataset_segment_conversion(df_segments_int):
    """Test that `TSDataset.to_dataset` makes casting of segment to string."""
    df = TSDataset.to_dataset(df_segments_int)
    assert np.all(df.columns.get_level_values("segment") == ["1", "2"])


def test_dataset_segment_conversion_during_init(df_segments_int):
    """Test that `TSDataset.__init__` makes casting of segment to string."""
    df = TSDataset.to_dataset(df_segments_int)
    # make conversion back to integers
    columns_frame = df.columns.to_frame()
    columns_frame["segment"] = columns_frame["segment"].astype(int)
    df.columns = pd.MultiIndex.from_frame(columns_frame)
    ts = TSDataset(df=df, freq="D")
    assert np.all(ts.columns.get_level_values("segment") == ["1", "2"])


@pytest.mark.xfail
def test_make_future_raise_error_on_diff_endings(ts_diff_endings):
    with pytest.raises(ValueError, match="All segments should end at the same timestamp"):
        ts_diff_endings.make_future(10)


def test_make_future_with_imputer(ts_diff_endings, ts_future):
    imputer = TimeSeriesImputerTransform(in_column="target")
    ts_diff_endings.fit_transform([imputer])
    future = ts_diff_endings.make_future(10, transforms=[imputer])
    assert_frame_equal(future.to_pandas(), ts_future.to_pandas())


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
    df, df_exog, known_future = df_and_regressors
    ts = TSDataset(df=df, df_exog=df_exog, freq="D", known_future=known_future)
    ts_future = ts.make_future(10)
    assert np.all(ts_future.index == pd.date_range(ts.index.max() + pd.Timedelta("1D"), periods=10, freq="D"))
    assert set(ts_future.columns.get_level_values("feature")) == {"target", "regressor_1", "regressor_2"}


@pytest.mark.parametrize("tail_steps", [11, 0])
def test_make_future_with_regressors_and_context(df_and_regressors, tail_steps):
    df, df_exog, known_future = df_and_regressors
    horizon = 10
    ts = TSDataset(df=df, df_exog=df_exog, freq="D", known_future=known_future)
    ts_future = ts.make_future(horizon, tail_steps=tail_steps)
    assert ts_future.index[tail_steps] == ts.index[-1] + pd.Timedelta("1 day")


def test_make_future_inherits_regressors(df_and_regressors):
    df, df_exog, known_future = df_and_regressors
    ts = TSDataset(df=df, df_exog=df_exog, freq="D", known_future=known_future)
    ts_future = ts.make_future(10)
    assert ts_future.regressors == ts.regressors


def test_make_future_inherits_hierarchy(product_level_constant_forecast_with_quantiles):
    ts = product_level_constant_forecast_with_quantiles
    future = ts.make_future(future_steps=2)
    assert future.hierarchical_structure is ts.hierarchical_structure


def test_make_future_removes_quantiles(product_level_constant_forecast_with_quantiles):
    ts = product_level_constant_forecast_with_quantiles
    future = ts.make_future(future_steps=2)
    assert len(future.target_quantiles_names) == 0


def test_make_future_removes_target_components(ts_with_target_components):
    ts = ts_with_target_components
    future = ts.make_future(future_steps=2)
    assert len(future.target_components_names) == 0


def test_make_future_warn_not_enough_regressors(df_and_regressors):
    """Check that warning is thrown if regressors don't have enough values for the future."""
    df, df_exog, known_future = df_and_regressors
    ts = TSDataset(df=df, df_exog=df_exog, freq="D", known_future=known_future)
    with pytest.warns(UserWarning, match="Some regressors don't have enough values"):
        ts.make_future(ts.df_exog.shape[0] + 100)


@pytest.mark.parametrize("exog_starts_later,exog_ends_earlier", ((True, False), (False, True), (True, True)))
def test_check_regressors_error(exog_starts_later: bool, exog_ends_earlier: bool):
    """Check that error is raised if regressors don't have enough values for the train data."""
    start_time_main = "2021-01-01"
    end_time_main = "2021-02-01"
    start_time_regressors = "2021-01-10" if exog_starts_later else start_time_main
    end_time_regressors = "2021-01-20" if exog_ends_earlier else end_time_main

    timestamp = pd.date_range("2021-01-01", "2021-02-01")
    df1 = pd.DataFrame({"timestamp": timestamp, "target": 11, "segment": "1"})
    df2 = pd.DataFrame({"timestamp": timestamp[5:], "target": 12, "segment": "2"})
    df = pd.concat([df1, df2], ignore_index=True)
    df = TSDataset.to_dataset(df)

    timestamp = pd.date_range(start_time_regressors, end_time_regressors)
    df1 = pd.DataFrame({"timestamp": timestamp, "regressor_aaa": 1, "segment": "1"})
    df2 = pd.DataFrame({"timestamp": timestamp[5:], "regressor_aaa": 2, "segment": "2"})
    df_regressors = pd.concat([df1, df2], ignore_index=True)
    df_regressors = TSDataset.to_dataset(df_regressors)

    with pytest.raises(ValueError):
        TSDataset._check_regressors(df=df, df_regressors=df_regressors)


def test_check_regressors_pass(df_and_regressors):
    """Check that regressors check on creation passes with correct regressors."""
    df, df_exog, _ = df_and_regressors
    _ = TSDataset._check_regressors(df=df, df_regressors=df_exog)


def test_check_regressors_pass_empty(df_and_regressors):
    """Check that regressors check on creation passes with no regressors."""
    df, _, _ = df_and_regressors
    _ = TSDataset._check_regressors(df=df, df_regressors=pd.DataFrame())


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


def test_finding_regressors_marked(df_and_regressors):
    """Check that ts.regressors property works correctly when regressors set."""
    df, df_exog, known_future = df_and_regressors
    ts = TSDataset(df=df, df_exog=df_exog, freq="D", known_future=["regressor_1", "regressor_2"])
    assert sorted(ts.regressors) == ["regressor_1", "regressor_2"]


def test_finding_regressors_unmarked(df_and_regressors):
    """Check that ts.regressors property works correctly when regressors don't set."""
    df, df_exog, _ = df_and_regressors
    ts = TSDataset(df=df, df_exog=df_exog, freq="D")
    assert sorted(ts.regressors) == []


def test_head_default(tsdf_with_exog):
    assert np.all(tsdf_with_exog.head() == tsdf_with_exog.df.head())


def test_tail_default(tsdf_with_exog):
    np.all(tsdf_with_exog.tail() == tsdf_with_exog.df.tail())


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


def test_to_flatten_simple(example_df):
    """Check that TSDataset.to_flatten works correctly in simple case."""
    flat_df = example_df
    sorted_columns = sorted(flat_df.columns)
    expected_df = flat_df[sorted_columns]
    obtained_df = TSDataset.to_flatten(TSDataset.to_dataset(flat_df))[sorted_columns]
    assert np.all(expected_df.columns == obtained_df.columns)
    assert np.all(expected_df.dtypes == obtained_df.dtypes)
    assert np.all(expected_df.values == obtained_df.values)


def test_to_flatten_with_exog(df_and_regressors_flat):
    """Check that TSDataset.to_flatten works correctly with exogenous features."""
    df, df_exog = df_and_regressors_flat

    # add boolean dtype
    df_exog["regressor_boolean"] = 1
    df_exog["regressor_boolean"] = df_exog["regressor_boolean"].astype("boolean")
    # add Int64 dtype
    df_exog["regressor_Int64"] = 1
    df_exog.loc[1, "regressor_Int64"] = None
    df_exog["regressor_Int64"] = df_exog["regressor_Int64"].astype("Int64")

    # construct expected result
    flat_df = pd.merge(left=df, right=df_exog, left_on=["timestamp", "segment"], right_on=["timestamp", "segment"])
    sorted_columns = sorted(flat_df.columns)
    expected_df = flat_df[sorted_columns]
    # add values to absent timestamps at one segment
    to_append = pd.DataFrame({"timestamp": df["timestamp"][:5], "segment": ["2"] * 5})
    dtypes = expected_df.dtypes.to_dict()
    expected_df = pd.concat((expected_df, to_append)).sort_values(by=["segment", "timestamp"]).reset_index(drop=True)
    # restore category dtypes: needed for old versions of pandas
    for column, dtype in dtypes.items():
        if dtype == "category":
            expected_df[column] = expected_df[column].astype(dtype)
    # this logic wouldn't work in general case, here we use that all features' names start with 'r'
    sorted_columns = ["timestamp", "segment", "target"] + sorted_columns[:-3]
    # reindex df to assert correct columns order
    expected_df = expected_df[sorted_columns]
    # get to_flatten result
    obtained_df = TSDataset.to_flatten(TSDataset.to_dataset(flat_df))
    pd.testing.assert_frame_equal(obtained_df, expected_df)


@pytest.mark.parametrize(
    "features, expected_columns",
    (
        ("all", ["timestamp", "target", "segment", "regressor_1", "regressor_2"]),
        (["regressor_2"], ["timestamp", "segment", "regressor_2"]),
    ),
)
def test_to_flatten_correct_columns(df_and_regressors, features, expected_columns):
    df, df_exog, known_future = df_and_regressors
    ts = TSDataset(df=df, df_exog=df_exog, freq="D", known_future=known_future)
    flattened_df = ts.to_flatten(ts.df, features=features)
    assert sorted(flattened_df.columns) == sorted(expected_columns)


def test_to_flatten_raise_error_incorrect_literal(df_and_regressors):
    df, df_exog, known_future = df_and_regressors
    ts = TSDataset(df=df, df_exog=df_exog, freq="D", known_future=known_future)
    with pytest.raises(ValueError, match="The only possible literal is 'all'"):
        _ = ts.to_flatten(ts.df, features="incorrect")


@pytest.mark.parametrize(
    "features, expected_columns",
    (
        ("all", ["target", "regressor_1", "regressor_2"]),
        (["regressor_2"], ["regressor_2"]),
    ),
)
def test_to_pandas_correct_columns(df_and_regressors, features, expected_columns):
    df, df_exog, known_future = df_and_regressors
    ts = TSDataset(df=df, df_exog=df_exog, freq="D", known_future=known_future)
    pandas_df = ts.to_pandas(flatten=False, features=features)
    got_columns = set(pandas_df.columns.get_level_values("feature"))
    assert sorted(got_columns) == sorted(expected_columns)


def test_to_pandas_raise_error_incorrect_literal(df_and_regressors):
    df, df_exog, known_future = df_and_regressors
    ts = TSDataset(df=df, df_exog=df_exog, freq="D", known_future=known_future)
    with pytest.raises(ValueError, match="The only possible literal is 'all'"):
        _ = ts.to_pandas(flatten=False, features="incorrect")


def test_transform_raise_warning_on_diff_endings(ts_diff_endings):
    with pytest.warns(Warning, match="Segments contains NaNs in the last timestamps."):
        ts_diff_endings.transform([])


def test_fit_transform_raise_warning_on_diff_endings(ts_diff_endings):
    with pytest.warns(Warning, match="Segments contains NaNs in the last timestamps."):
        ts_diff_endings.fit_transform([])


def test_gather_common_data(ts_info):
    """Check that TSDataset._gather_common_data correctly finds common data for info/describe methods."""
    common_data = ts_info._gather_common_data()
    assert common_data["num_segments"] == 3
    assert common_data["num_exogs"] == 2
    assert common_data["num_regressors"] == 2
    assert common_data["num_known_future"] == 2
    assert common_data["freq"] == "D"


def test_gather_segments_data(ts_info):
    """Check that TSDataset._gather_segments_data correctly finds segment data for info/describe methods."""
    segments_dict = ts_info._gather_segments_data(ts_info.segments)
    segment_df = pd.DataFrame(segments_dict, index=ts_info.segments)

    assert segment_df.loc["1", "start_timestamp"] == pd.Timestamp("2021-01-01")
    assert segment_df.loc["2", "start_timestamp"] == pd.Timestamp("2021-01-06")
    assert segment_df.loc["3", "start_timestamp"] is pd.NaT
    assert segment_df.loc["1", "end_timestamp"] == pd.Timestamp("2021-02-01")
    assert segment_df.loc["2", "end_timestamp"] == pd.Timestamp("2021-01-29")
    assert segment_df.loc["3", "end_timestamp"] is pd.NaT
    assert segment_df.loc["1", "length"] == 32
    assert segment_df.loc["2", "length"] == 24
    assert segment_df.loc["3", "length"] is pd.NA
    assert segment_df.loc["1", "num_missing"] == 1
    assert segment_df.loc["2", "num_missing"] == 0
    assert segment_df.loc["3", "num_missing"] is pd.NA


def test_describe(ts_info):
    """Check that TSDataset.describe works correctly."""
    description = ts_info.describe()

    assert np.all(description.index == ts_info.segments)
    assert description.loc["1", "start_timestamp"] == pd.Timestamp("2021-01-01")
    assert description.loc["2", "start_timestamp"] == pd.Timestamp("2021-01-06")
    assert description.loc["3", "start_timestamp"] is pd.NaT
    assert description.loc["1", "end_timestamp"] == pd.Timestamp("2021-02-01")
    assert description.loc["2", "end_timestamp"] == pd.Timestamp("2021-01-29")
    assert description.loc["3", "end_timestamp"] is pd.NaT
    assert description.loc["1", "length"] == 32
    assert description.loc["2", "length"] == 24
    assert description.loc["3", "length"] is pd.NA
    assert description.loc["1", "num_missing"] == 1
    assert description.loc["2", "num_missing"] == 0
    assert description.loc["3", "num_missing"] is pd.NA
    assert np.all(description["num_segments"] == 3)
    assert np.all(description["num_exogs"] == 2)
    assert np.all(description["num_regressors"] == 2)
    assert np.all(description["num_known_future"] == 2)
    assert np.all(description["freq"] == "D")


@pytest.fixture()
def ts_with_regressors(df_and_regressors):
    df, df_exog, regressors = df_and_regressors
    ts = TSDataset(df=df, freq="D", df_exog=df_exog, known_future="all")
    return ts


def test_to_dataset_not_modify_dataframe():
    timestamp = pd.date_range("2021-01-01", "2021-02-01")
    df_original = pd.DataFrame({"timestamp": timestamp, "target": 11, "segment": 1})
    df_copy = df_original.copy(deep=True)
    df_mod = TSDataset.to_dataset(df_original)
    pd.testing.assert_frame_equal(df_original, df_copy)


@pytest.mark.parametrize("start_idx,end_idx", [(1, None), (None, 1), (1, 2), (1, -1)])
def test_tsdataset_idx_slice(tsdf_with_exog, start_idx, end_idx):
    ts_slice = tsdf_with_exog.tsdataset_idx_slice(start_idx=start_idx, end_idx=end_idx)
    assert ts_slice.known_future == tsdf_with_exog.known_future
    assert ts_slice.regressors == tsdf_with_exog.regressors
    pd.testing.assert_frame_equal(ts_slice.df, tsdf_with_exog.df.iloc[start_idx:end_idx])
    pd.testing.assert_frame_equal(ts_slice.df_exog, tsdf_with_exog.df_exog)


def test_tsdataset_idx_slice_pass_target_components_to_output(ts_with_target_components):
    ts_slice = ts_with_target_components.tsdataset_idx_slice(start_idx=1, end_idx=2)
    assert sorted(ts_slice.target_components_names) == sorted(ts_with_target_components.target_components_names)


def test_to_torch_dataset_without_drop(tsdf_with_exog):
    def make_samples(df):
        return [{"target": df.target.values, "segment": df["segment"].values[0]}]

    torch_dataset = tsdf_with_exog.to_torch_dataset(make_samples, dropna=False)
    assert len(torch_dataset) == len(tsdf_with_exog.segments)
    np.testing.assert_array_equal(
        torch_dataset[0]["target"], tsdf_with_exog.df.loc[:, pd.IndexSlice["Moscow", "target"]].values
    )
    np.testing.assert_array_equal(
        torch_dataset[1]["target"], tsdf_with_exog.df.loc[:, pd.IndexSlice["Omsk", "target"]].values
    )


def test_to_torch_dataset_with_drop(tsdf_with_exog):
    def make_samples(df):
        return [{"target": df.target.values, "segment": df["segment"].values[0]}]

    fill_na_idx = tsdf_with_exog.df.index[3]
    tsdf_with_exog.df.loc[:fill_na_idx, pd.IndexSlice["Moscow", "target"]] = np.nan

    torch_dataset = tsdf_with_exog.to_torch_dataset(make_samples, dropna=True)
    assert len(torch_dataset) == len(tsdf_with_exog.segments)
    np.testing.assert_array_equal(
        torch_dataset[0]["target"],
        tsdf_with_exog.df.loc[fill_na_idx + pd.Timedelta("1 day") :, pd.IndexSlice["Moscow", "target"]].values,
    )
    np.testing.assert_array_equal(
        torch_dataset[1]["target"], tsdf_with_exog.df.loc[:, pd.IndexSlice["Omsk", "target"]].values
    )


def test_add_columns_from_pandas_update_df(df_and_regressors, df_update_add_column, df_updated_add_column):
    df, _, _ = df_and_regressors
    ts = TSDataset(df=df, freq="D")
    ts.add_columns_from_pandas(df_update=df_update_add_column, update_exog=False)
    pd.testing.assert_frame_equal(ts.df, df_updated_add_column)


def test_add_columns_from_pandas_update_df_exog(df_and_regressors, df_update_add_column, df_exog_updated_add_column):
    df, df_exog, _ = df_and_regressors
    ts = TSDataset(df=df, freq="D", df_exog=df_exog)
    ts.add_columns_from_pandas(df_update=df_update_add_column, update_exog=True)
    pd.testing.assert_frame_equal(ts.df_exog, df_exog_updated_add_column)


@pytest.mark.parametrize(
    "known_future, regressors, expected_regressors",
    (
        ([], ["regressor_1"], ["regressor_1"]),
        (["regressor_1"], ["regressor_1", "regressor_2"], ["regressor_1", "regressor_2"]),
    ),
)
def test_add_columns_from_pandas_update_regressors(
    df_and_regressors, df_update_add_column, known_future, regressors, expected_regressors
):
    df, df_exog, _ = df_and_regressors
    ts = TSDataset(df=df, freq="D", df_exog=df_exog, known_future=known_future)
    ts.add_columns_from_pandas(df_update=df_update_add_column, update_exog=True, regressors=regressors)
    assert sorted(ts.regressors) == sorted(expected_regressors)


def test_update_columns_from_pandas(df_and_regressors, df_update_update_column, df_updated_update_column):
    df, _, _ = df_and_regressors
    ts = TSDataset(df=df, freq="D")
    ts.update_columns_from_pandas(df_update=df_update_update_column)
    pd.testing.assert_frame_equal(ts.df, df_updated_update_column)


@pytest.mark.filterwarnings("ignore: Features {'out_of_dataset_column'} are not present in")
@pytest.mark.parametrize(
    "features, drop_from_exog, df_expected_columns, df_exog_expected_columns",
    (
        (
            ["regressor_2"],
            False,
            ["timestamp", "segment", "target", "regressor_1"],
            ["timestamp", "segment", "regressor_1", "regressor_2"],
        ),
        (
            ["regressor_2"],
            True,
            ["timestamp", "segment", "target", "regressor_1"],
            ["timestamp", "segment", "regressor_1"],
        ),
        (
            ["regressor_2", "out_of_dataset_column"],
            True,
            ["timestamp", "segment", "target", "regressor_1"],
            ["timestamp", "segment", "regressor_1"],
        ),
    ),
)
def test_drop_features(df_and_regressors, features, drop_from_exog, df_expected_columns, df_exog_expected_columns):
    df, df_exog, known_future = df_and_regressors
    ts = TSDataset(df=df, df_exog=df_exog, freq="D", known_future=known_future)
    ts.drop_features(features=features, drop_from_exog=drop_from_exog)
    df_columns, df_exog_columns = ts.to_flatten(ts.df).columns, ts.to_flatten(ts.df_exog).columns
    assert sorted(df_columns) == sorted(df_expected_columns)
    assert sorted(df_exog_columns) == sorted(df_exog_expected_columns)


def test_drop_features_raise_warning_on_unknown_columns(
    df_and_regressors, features=["regressor_2", "out_of_dataset_column"]
):
    df, df_exog, known_future = df_and_regressors
    ts = TSDataset(df=df, df_exog=df_exog, freq="D", known_future=known_future)
    with pytest.warns(UserWarning, match="Features {'out_of_dataset_column'} are not present in df!"):
        ts.drop_features(features=features, drop_from_exog=False)


@pytest.mark.filterwarnings("ignore: Features {'out_of_dataset_column'} are not present in")
@pytest.mark.parametrize(
    "features, expected_regressors",
    (
        (["target", "regressor_2"], ["regressor_1"]),
        (["out_of_dataset_column"], ["regressor_1", "regressor_2"]),
    ),
)
def test_drop_features_update_regressors(df_and_regressors, features, expected_regressors):
    df, df_exog, known_future = df_and_regressors
    ts = TSDataset(df=df, df_exog=df_exog, freq="D", known_future=known_future)
    ts.drop_features(features=features, drop_from_exog=False)
    assert sorted(ts.regressors) == sorted(expected_regressors)


def test_drop_features_throw_error_on_target_components(ts_with_target_components):
    with pytest.raises(
        ValueError,
        match="Target components can't be dropped from the dataset using this method! Use `drop_target_components` method!",
    ):
        ts_with_target_components.drop_features(features=ts_with_target_components.target_components_names)


def test_get_target_components_on_dataset_without_components(example_tsds):
    target_components_df = example_tsds.get_target_components()
    assert target_components_df is None


def test_get_target_components(
    ts_with_target_components, expected_components=["target_component_a", "target_component_b"]
):
    expected_target_components_df = ts_with_target_components.to_pandas(features=expected_components)
    target_components_df = ts_with_target_components.get_target_components()
    pd.testing.assert_frame_equal(target_components_df, expected_target_components_df)


def test_add_target_components_throw_error_adding_components_second_time(
    ts_with_target_components, target_components_df
):
    with pytest.raises(ValueError, match="Dataset already contains target components!"):
        ts_with_target_components.add_target_components(target_components_df=target_components_df)


@pytest.mark.parametrize(
    "inconsistent_target_components_names_fixture",
    [("inconsistent_target_components_names_df"), ("inconsistent_target_components_names_duplication_df")],
)
def test_add_target_components_throw_error_inconsistent_components_names(
    ts_without_target_components, inconsistent_target_components_names_fixture, request
):
    inconsistent_target_components_names_df = request.getfixturevalue(inconsistent_target_components_names_fixture)
    with pytest.raises(ValueError, match="Set of target components differs between segments '1' and '2'!"):
        ts_without_target_components.add_target_components(target_components_df=inconsistent_target_components_names_df)


def test_add_target_components_throw_error_inconsistent_components_values(
    ts_without_target_components, inconsistent_target_components_values_df
):
    with pytest.raises(ValueError, match="Components don't sum up to target!"):
        ts_without_target_components.add_target_components(
            target_components_df=inconsistent_target_components_values_df
        )


def test_add_target_components(ts_without_target_components, ts_with_target_components, target_components_df):
    ts_without_target_components.add_target_components(target_components_df=target_components_df)
    pd.testing.assert_frame_equal(ts_without_target_components.to_pandas(), ts_with_target_components.to_pandas())


def test_drop_target_components(ts_with_target_components, ts_without_target_components):
    ts_with_target_components.drop_target_components()
    assert ts_with_target_components.target_components_names == ()
    pd.testing.assert_frame_equal(
        ts_with_target_components.to_pandas(),
        ts_without_target_components.to_pandas(),
    )


def test_drop_target_components_without_components_in_dataset(ts_without_target_components):
    ts_without_target_components.drop_target_components()
    assert ts_without_target_components.target_components_names == ()


def test_inverse_transform_target_components(ts_with_target_components, inverse_transformed_components_df):
    transform = AddConstTransform(in_column="target", value=-10)
    transform.fit(ts=ts_with_target_components)
    ts_with_target_components.inverse_transform([transform])
    assert sorted(ts_with_target_components.target_components_names) == sorted(
        set(inverse_transformed_components_df.columns.get_level_values("feature"))
    )
    pd.testing.assert_frame_equal(ts_with_target_components.get_target_components(), inverse_transformed_components_df)


def test_inverse_transform_with_target_components_fails_keep_target_components(ts_with_target_components):
    transform = DifferencingTransform(in_column="target")
    with suppress(ValueError):
        ts_with_target_components.inverse_transform(transforms=[transform])
    assert len(ts_with_target_components.target_components_names) > 0


@pytest.mark.parametrize(
    "fixture_name, expected_quantiles",
    (("example_tsds", ()), ("product_level_constant_forecast_with_quantiles", ("target_0.25", "target_0.75"))),
)
def test_get_target_quantiles_names(fixture_name, expected_quantiles, request):
    ts = request.getfixturevalue(fixture_name)
    target_quantiles_names = ts.target_quantiles_names
    assert sorted(target_quantiles_names) == sorted(expected_quantiles)
