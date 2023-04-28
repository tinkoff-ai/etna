from copy import deepcopy
from typing import List

import numpy as np
import pandas as pd
import pytest

from etna.datasets import TSDataset
from etna.datasets import generate_const_df
from etna.transforms import BoxCoxTransform
from etna.transforms import MaxAbsScalerTransform
from etna.transforms import MinMaxScalerTransform
from etna.transforms import RobustScalerTransform
from etna.transforms import StandardScalerTransform
from etna.transforms import YeoJohnsonTransform
from tests.utils import select_segments_subset


@pytest.fixture
def multicolumn_ts(random_seed):
    df = generate_const_df(start_time="2020-01-01", periods=20, freq="D", scale=1.0, n_segments=3)
    df["target"] += np.random.uniform(0, 0.1, size=df.shape[0])
    df_exog = df.copy().rename(columns={"target": "exog_1"})
    for i in range(2, 6):
        df_exog[f"exog_{i}"] = float(i) + np.random.uniform(0, 0.1, size=df.shape[0])

    df_formatted = TSDataset.to_dataset(df)
    df_exog_formatted = TSDataset.to_dataset(df_exog)

    return TSDataset(df=df_formatted, df_exog=df_exog_formatted, freq="D")


def extract_new_features_columns(transformed_df: pd.DataFrame, initial_df: pd.DataFrame) -> List[str]:
    """Extract columns from feature level that are present in transformed_df but not present in initial_df."""
    return (
        transformed_df.columns.get_level_values("feature")
        .difference(initial_df.columns.get_level_values("feature"))
        .unique()
        .tolist()
    )


@pytest.mark.parametrize(
    "transform_constructor",
    (
        BoxCoxTransform,
        YeoJohnsonTransform,
        StandardScalerTransform,
        RobustScalerTransform,
        MinMaxScalerTransform,
        MaxAbsScalerTransform,
        StandardScalerTransform,
        RobustScalerTransform,
        MinMaxScalerTransform,
    ),
)
def test_fail_invalid_mode(transform_constructor):
    """Test that transform raises error in invalid mode."""
    with pytest.raises(ValueError):
        _ = transform_constructor(mode="non_existent")


@pytest.mark.parametrize(
    "transform_constructor",
    (
        BoxCoxTransform,
        YeoJohnsonTransform,
        StandardScalerTransform,
        RobustScalerTransform,
        MinMaxScalerTransform,
        MaxAbsScalerTransform,
        StandardScalerTransform,
        RobustScalerTransform,
        MinMaxScalerTransform,
    ),
)
def test_warning_not_inplace(transform_constructor):
    """Test that transform raises warning if inplace is set to True, but out_column is also given."""
    with pytest.warns(UserWarning, match="Transformation will be applied inplace"):
        _ = transform_constructor(inplace=True, out_column="new_exog")


@pytest.mark.parametrize(
    "transform_constructor",
    [
        BoxCoxTransform,
        YeoJohnsonTransform,
        StandardScalerTransform,
        RobustScalerTransform,
        MinMaxScalerTransform,
        MaxAbsScalerTransform,
        StandardScalerTransform,
        RobustScalerTransform,
        MinMaxScalerTransform,
    ],
)
@pytest.mark.parametrize(
    "in_column",
    [
        "exog_1",
        ["exog_1", "exog_2"],
    ],
)
def test_inplace_no_new_columns(transform_constructor, in_column, multicolumn_ts):
    """Test that transform in inplace mode doesn't generate new columns."""
    transform = transform_constructor(in_column=in_column, inplace=True)
    initial_df = multicolumn_ts.to_pandas()
    transformed_df = transform.fit_transform(multicolumn_ts).to_pandas()

    # check new columns
    new_columns = extract_new_features_columns(transformed_df, initial_df)
    assert len(new_columns) == 0

    # check that output columns are input columns
    assert transform.out_columns == transform.in_column


@pytest.mark.parametrize(
    "transform_constructor",
    [
        BoxCoxTransform,
        YeoJohnsonTransform,
        StandardScalerTransform,
        RobustScalerTransform,
        MinMaxScalerTransform,
        MaxAbsScalerTransform,
        StandardScalerTransform,
        RobustScalerTransform,
        MinMaxScalerTransform,
    ],
)
@pytest.mark.parametrize(
    "in_column",
    [
        "exog_1",
        ["exog_1", "exog_2"],
    ],
)
def test_creating_columns(transform_constructor, in_column, multicolumn_ts):
    """Test that transform creates new columns according to out_column parameter."""
    transform = transform_constructor(in_column=in_column, out_column="new_exog", inplace=False)
    initial_df = multicolumn_ts.to_pandas()
    transformed_df = transform.fit_transform(multicolumn_ts).to_pandas()

    # check new columns
    new_columns = set(extract_new_features_columns(transformed_df, initial_df))
    in_column = [in_column] if isinstance(in_column, str) else in_column
    expected_columns = {f"new_exog_{column}" for column in in_column}
    assert new_columns == expected_columns

    # check that output columns are matching input columns
    assert len(transform.in_column) == len(transform.out_columns)
    assert all(
        f"new_exog_{column}" == new_column for column, new_column in zip(transform.in_column, transform.out_columns)
    )


@pytest.mark.parametrize(
    "transform_constructor",
    [
        BoxCoxTransform,
        YeoJohnsonTransform,
        StandardScalerTransform,
        RobustScalerTransform,
        MinMaxScalerTransform,
        MaxAbsScalerTransform,
        StandardScalerTransform,
        RobustScalerTransform,
        MinMaxScalerTransform,
    ],
)
@pytest.mark.parametrize(
    "in_column",
    [
        "exog_1",
        ["exog_1", "exog_2"],
    ],
)
def test_generated_column_names(transform_constructor, in_column, multicolumn_ts):
    """Test that transform generates names for the columns correctly."""
    transform = transform_constructor(in_column=in_column, out_column=None, inplace=False)
    initial_df = multicolumn_ts.to_pandas()
    transformed_df = transform.fit_transform(deepcopy(multicolumn_ts)).to_pandas()
    segments = sorted(multicolumn_ts.segments)

    new_columns = extract_new_features_columns(transformed_df, initial_df)
    # check new columns
    for column in new_columns:
        # create transform from column
        transform_temp = eval(column)
        df_temp = transform_temp.fit_transform(deepcopy(multicolumn_ts)).to_pandas()
        columns_temp = extract_new_features_columns(df_temp, initial_df)
        # compare column names and column values
        assert len(columns_temp) == 1
        column_temp = columns_temp[0]
        assert column_temp == column
        assert np.all(
            df_temp.loc[:, pd.IndexSlice[segments, column_temp]]
            == transformed_df.loc[:, pd.IndexSlice[segments, column]]
        )

    # check that output columns are matching input columns
    assert len(transform.in_column) == len(transform.out_columns)
    # check that name if this input column is present inside name of this output column
    assert all((column in new_column) for column, new_column in zip(transform.in_column, transform.out_columns))


@pytest.mark.parametrize(
    "transform_constructor",
    [
        BoxCoxTransform,
        YeoJohnsonTransform,
        StandardScalerTransform,
        RobustScalerTransform,
        MinMaxScalerTransform,
        MaxAbsScalerTransform,
        StandardScalerTransform,
        RobustScalerTransform,
        MinMaxScalerTransform,
    ],
)
def test_all_columns(transform_constructor, multicolumn_ts):
    """Test that transform can process all columns using None value for in_column."""
    transform = transform_constructor(in_column=None, out_column=None, inplace=False)
    initial_df = multicolumn_ts.df.copy()
    transformed_df = transform.fit_transform(multicolumn_ts).to_pandas()

    new_columns = extract_new_features_columns(transformed_df, initial_df)
    assert len(new_columns) == initial_df.columns.get_level_values("feature").nunique()


@pytest.mark.parametrize(
    "transform_constructor",
    [
        BoxCoxTransform,
        YeoJohnsonTransform,
        StandardScalerTransform,
        RobustScalerTransform,
        MinMaxScalerTransform,
        MaxAbsScalerTransform,
        StandardScalerTransform,
        RobustScalerTransform,
        MinMaxScalerTransform,
    ],
)
@pytest.mark.parametrize(
    "in_column", [["exog_1", "exog_2", "exog_3"], ["exog_2", "exog_1", "exog_3"], ["exog_3", "exog_2", "exog_1"]]
)
@pytest.mark.parametrize(
    "mode",
    [
        "macro",
        "per-segment",
    ],
)
def test_ordering(transform_constructor, in_column, mode, multicolumn_ts):
    """Test that transform don't mix columns between each other."""
    transform = transform_constructor(in_column=in_column, out_column=None, mode=mode, inplace=False)
    transforms_one_column = [
        transform_constructor(in_column=column, out_column=None, mode=mode, inplace=False) for column in in_column
    ]

    segments = sorted(multicolumn_ts.segments)
    transformed_df = transform.fit_transform(deepcopy(multicolumn_ts)).to_pandas()

    transformed_dfs_one_column = []
    for transform_one_column in transforms_one_column:
        transformed_dfs_one_column.append(transform_one_column.fit_transform(deepcopy(multicolumn_ts)))

    in_to_out_columns = dict(zip(transform.in_column, transform.out_columns))
    for i, column in enumerate(in_column):
        # find relevant column name in transformed_df
        column_multi = in_to_out_columns[column]

        # find relevant column name in transformed_dfs_one_column[i]
        column_single = transforms_one_column[i].out_columns[0]

        df_multi = transformed_df.loc[:, pd.IndexSlice[segments, column_multi]]
        df_single = transformed_dfs_one_column[i].loc[:, pd.IndexSlice[segments, column_single]]
        assert np.all(df_multi.values == df_single.values)


@pytest.mark.parametrize(
    "transform_constructor",
    [
        BoxCoxTransform,
        YeoJohnsonTransform,
        StandardScalerTransform,
        RobustScalerTransform,
        MinMaxScalerTransform,
        MaxAbsScalerTransform,
        StandardScalerTransform,
        RobustScalerTransform,
        MinMaxScalerTransform,
    ],
)
def test_get_regressors_info_not_fitted(transform_constructor):
    transform = transform_constructor(in_column="target")
    with pytest.raises(ValueError, match="Fit the transform to get the correct regressors info!"):
        _ = transform.get_regressors_info()


@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize(
    "in_column",
    [
        "exog_1",
        ["exog_1", "exog_2"],
    ],
)
@pytest.mark.parametrize(
    "mode",
    [
        "macro",
        "per-segment",
    ],
)
@pytest.mark.parametrize(
    "transform_constructor",
    [
        BoxCoxTransform,
        YeoJohnsonTransform,
        StandardScalerTransform,
        RobustScalerTransform,
        MinMaxScalerTransform,
        MaxAbsScalerTransform,
        StandardScalerTransform,
        RobustScalerTransform,
        MinMaxScalerTransform,
    ],
)
def test_transform_not_fitted_fail(transform_constructor, mode, in_column, inplace, multicolumn_ts):
    transform = transform_constructor(mode=mode, in_column=in_column, inplace=inplace)

    with pytest.raises(ValueError, match="The transform isn't fitted"):
        _ = transform.transform(multicolumn_ts)


@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize(
    "in_column",
    [
        "exog_1",
        ["exog_1", "exog_2"],
    ],
)
@pytest.mark.parametrize(
    "mode",
    [
        "macro",
        "per-segment",
    ],
)
@pytest.mark.parametrize(
    "transform_constructor",
    [
        BoxCoxTransform,
        YeoJohnsonTransform,
        StandardScalerTransform,
        RobustScalerTransform,
        MinMaxScalerTransform,
        MaxAbsScalerTransform,
        StandardScalerTransform,
        RobustScalerTransform,
        MinMaxScalerTransform,
    ],
)
def test_inverse_transform_not_fitted_fail(transform_constructor, mode, in_column, inplace, multicolumn_ts):
    transform = transform_constructor(mode=mode, in_column=in_column, inplace=inplace)

    with pytest.raises(ValueError, match="The transform isn't fitted"):
        _ = transform.inverse_transform(multicolumn_ts)


def _check_same_segments(df_1: pd.DataFrame, df_2: pd.DataFrame):
    df_1_segments = set(df_1.columns.get_level_values("segment"))
    df_2_segments = set(df_2.columns.get_level_values("segment"))
    assert df_1_segments == df_2_segments


@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize(
    "in_column",
    [
        "exog_1",
        ["exog_1", "exog_2"],
    ],
)
@pytest.mark.parametrize(
    "mode",
    [
        "macro",
        "per-segment",
    ],
)
@pytest.mark.parametrize(
    "transform_constructor",
    [
        BoxCoxTransform,
        YeoJohnsonTransform,
        StandardScalerTransform,
        RobustScalerTransform,
        MinMaxScalerTransform,
        MaxAbsScalerTransform,
        StandardScalerTransform,
        RobustScalerTransform,
        MinMaxScalerTransform,
    ],
)
def test_transform_subset_segments(transform_constructor, mode, in_column, inplace, multicolumn_ts):
    train_ts = multicolumn_ts
    test_ts = select_segments_subset(ts=multicolumn_ts, segments=["segment_0", "segment_2"])
    test_df = test_ts.to_pandas()
    transform = transform_constructor(mode=mode, in_column=in_column, inplace=inplace)

    transform.fit(train_ts)
    transformed_df = transform.transform(test_ts).to_pandas()

    _check_same_segments(transformed_df, test_df)


@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize(
    "in_column",
    [
        "exog_1",
        ["exog_1", "exog_2"],
    ],
)
@pytest.mark.parametrize(
    "mode",
    [
        "macro",
        "per-segment",
    ],
)
@pytest.mark.parametrize(
    "transform_constructor",
    [
        BoxCoxTransform,
        YeoJohnsonTransform,
        StandardScalerTransform,
        RobustScalerTransform,
        MinMaxScalerTransform,
        MaxAbsScalerTransform,
        StandardScalerTransform,
        RobustScalerTransform,
        MinMaxScalerTransform,
    ],
)
def test_inverse_transform_subset_segments(transform_constructor, mode, in_column, inplace, multicolumn_ts):
    train_ts = multicolumn_ts
    test_ts = select_segments_subset(ts=multicolumn_ts, segments=["segment_0", "segment_2"])
    test_df = test_ts.to_pandas()
    transform = transform_constructor(mode=mode, in_column=in_column, inplace=inplace)

    transform.fit(train_ts)
    inv_transformed_df = transform.inverse_transform(test_ts).to_pandas()

    _check_same_segments(inv_transformed_df, test_df)


@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize(
    "in_column",
    [
        "exog_1",
        ["exog_1", "exog_2"],
    ],
)
@pytest.mark.parametrize(
    "transform_constructor",
    [
        BoxCoxTransform,
        YeoJohnsonTransform,
        StandardScalerTransform,
        RobustScalerTransform,
        MinMaxScalerTransform,
        MaxAbsScalerTransform,
        StandardScalerTransform,
        RobustScalerTransform,
        MinMaxScalerTransform,
    ],
)
def test_transform_new_segments_macro(transform_constructor, in_column, inplace, multicolumn_ts):
    train_ts = select_segments_subset(ts=multicolumn_ts, segments=["segment_0", "segment_1"])
    test_ts = select_segments_subset(ts=multicolumn_ts, segments=["segment_2"])
    test_df = test_ts.to_pandas()
    transform = transform_constructor(mode="macro", in_column=in_column, inplace=inplace)

    transform.fit(train_ts)
    transformed_df = transform.transform(test_ts).to_pandas()

    _check_same_segments(transformed_df, test_df)


@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize(
    "in_column",
    [
        "exog_1",
        ["exog_1", "exog_2"],
    ],
)
@pytest.mark.parametrize(
    "transform_constructor",
    [
        BoxCoxTransform,
        YeoJohnsonTransform,
        StandardScalerTransform,
        RobustScalerTransform,
        MinMaxScalerTransform,
        MaxAbsScalerTransform,
        StandardScalerTransform,
        RobustScalerTransform,
        MinMaxScalerTransform,
    ],
)
def test_transform_new_segments_per_segment_fail(transform_constructor, in_column, inplace, multicolumn_ts):
    train_ts = select_segments_subset(ts=multicolumn_ts, segments=["segment_0", "segment_1"])
    test_ts = select_segments_subset(ts=multicolumn_ts, segments=["segment_2"])
    transform = transform_constructor(mode="per-segment", in_column=in_column, inplace=inplace)

    transform.fit(train_ts)
    with pytest.raises(
        NotImplementedError, match="This transform can't process segments that weren't present on train data"
    ):
        _ = transform.transform(test_ts)


@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize(
    "in_column",
    [
        "exog_1",
        ["exog_1", "exog_2"],
    ],
)
@pytest.mark.parametrize(
    "transform_constructor",
    [
        BoxCoxTransform,
        YeoJohnsonTransform,
        StandardScalerTransform,
        RobustScalerTransform,
        MinMaxScalerTransform,
        MaxAbsScalerTransform,
        StandardScalerTransform,
        RobustScalerTransform,
        MinMaxScalerTransform,
    ],
)
def test_inverse_transform_new_segments_macro(transform_constructor, in_column, inplace, multicolumn_ts):
    train_ts = select_segments_subset(ts=multicolumn_ts, segments=["segment_0", "segment_1"])
    test_ts = select_segments_subset(ts=multicolumn_ts, segments=["segment_2"])
    test_df = test_ts.to_pandas()
    transform = transform_constructor(mode="macro", in_column=in_column, inplace=inplace)

    transform.fit(train_ts)
    transformed_df = transform.inverse_transform(test_ts).to_pandas()

    _check_same_segments(transformed_df, test_df)


@pytest.mark.parametrize(
    "in_column",
    [
        "exog_1",
        ["exog_1", "exog_2"],
    ],
)
@pytest.mark.parametrize(
    "transform_constructor",
    [
        BoxCoxTransform,
        YeoJohnsonTransform,
        StandardScalerTransform,
        RobustScalerTransform,
        MinMaxScalerTransform,
        MaxAbsScalerTransform,
        StandardScalerTransform,
        RobustScalerTransform,
        MinMaxScalerTransform,
    ],
)
def test_inverse_transform_new_segments_per_segment_non_inplace(transform_constructor, in_column, multicolumn_ts):
    train_ts = select_segments_subset(ts=multicolumn_ts, segments=["segment_0", "segment_1"])
    test_ts = select_segments_subset(ts=multicolumn_ts, segments=["segment_2"])
    test_df = test_ts.to_pandas()
    transform = transform_constructor(mode="per-segment", in_column=in_column, inplace=False)

    transform.fit(train_ts)
    inv_transformed_df = transform.inverse_transform(test_ts).to_pandas()

    pd.testing.assert_frame_equal(inv_transformed_df, test_df)


@pytest.mark.parametrize(
    "in_column",
    [
        "exog_1",
        ["exog_1", "exog_2"],
    ],
)
@pytest.mark.parametrize(
    "transform_constructor",
    [
        BoxCoxTransform,
        YeoJohnsonTransform,
        StandardScalerTransform,
        RobustScalerTransform,
        MinMaxScalerTransform,
        MaxAbsScalerTransform,
        StandardScalerTransform,
        RobustScalerTransform,
        MinMaxScalerTransform,
    ],
)
def test_inverse_transform_new_segments_per_segment_inplace_fail(transform_constructor, in_column, multicolumn_ts):
    train_ts = select_segments_subset(ts=multicolumn_ts, segments=["segment_0", "segment_1"])
    test_ts = select_segments_subset(ts=multicolumn_ts, segments=["segment_2"])
    transform = transform_constructor(mode="per-segment", in_column=in_column, inplace=True)

    transform.fit(train_ts)
    with pytest.raises(
        NotImplementedError, match="This transform can't process segments that weren't present on train data"
    ):
        _ = transform.inverse_transform(test_ts)
