import numpy as np
import pandas as pd
import pytest

from etna.datasets import TSDataset
from etna.datasets import generate_ar_df
from etna.transforms import AddConstTransform
from etna.transforms import LagTransform
from etna.transforms import LambdaTransform
from etna.transforms import LogTransform


@pytest.fixture
def ts_non_negative():
    df = generate_ar_df(
        start_time="2020-01-01", periods=300, ar_coef=[1], sigma=1, n_segments=3, random_seed=0, freq="D"
    )
    df = TSDataset.to_dataset(df)
    df = df.apply(lambda x: np.abs(x))
    ts = TSDataset(df, freq="D")
    return ts


@pytest.fixture
def ts_range_const():
    periods = 100
    df_1 = pd.DataFrame(
        {"timestamp": pd.date_range("2022-06-22", periods=periods), "target": np.arange(0, periods), "segment": 1}
    )
    df_2 = pd.DataFrame(
        {
            "timestamp": pd.date_range("2022-06-22", periods=periods),
            "target": np.array([1, 3] * (periods // 2)),
            "segment": 2,
        }
    )
    df = pd.concat([df_1, df_2])
    df = TSDataset.to_dataset(df)
    ts = TSDataset(df, freq="D")
    return ts


@pytest.mark.parametrize(
    "transform_original, transform_function, out_column",
    [
        (
            LogTransform(in_column="target", out_column="transform_target", inplace=False),
            lambda x: np.log10(x + 1),
            "transform_target",
        ),
        (
            AddConstTransform(in_column="target", out_column="transform_target", value=1, inplace=False),
            lambda x: x + 1,
            "transform_target",
        ),
        (
            LagTransform(in_column="target", out_column="transform_target", lags=[1]),
            lambda x: x.shift(1),
            "transform_target_1",
        ),
    ],
)
def test_save_transform(ts_non_negative, transform_original, transform_function, out_column):
    ts_copy = TSDataset(ts_non_negative.to_pandas(), freq="D")
    ts_copy.fit_transform([transform_original])
    ts = ts_non_negative
    ts.fit_transform(
        [LambdaTransform(in_column="target", out_column=out_column, transform_func=transform_function, inplace=False)]
    )
    assert set(ts_copy.columns) == set(ts.columns)
    for column in ts.columns:
        np.testing.assert_allclose(ts_copy[:, :, column], ts[:, :, column], rtol=1e-9)


def test_nesessary_inverse_transform(ts_non_negative):
    with pytest.raises(ValueError, match="inverse_transform_func must be defined, when inplace=True"):
        transform = LambdaTransform(in_column="target", inplace=True, transform_func=lambda x: x)
        ts_non_negative.fit_transform([transform])


def test_interface_inplace(ts_non_negative):
    transform = LambdaTransform(
        in_column="target", inplace=True, transform_func=lambda x: x, inverse_transform_func=lambda x: x
    )
    original_columns = set(ts_non_negative.columns)
    ts_non_negative.fit_transform([transform])
    assert set(ts_non_negative.columns) == original_columns
    ts_non_negative.inverse_transform()
    assert set(ts_non_negative.columns) == original_columns


def test_interface_not_inplace(ts_non_negative):
    add_column = "target_transformed"
    transform = LambdaTransform(in_column="target", out_column=add_column, transform_func=lambda x: x, inplace=False)
    original_columns = set(ts_non_negative.columns)
    ts_non_negative.fit_transform([transform])
    assert set(ts_non_negative.columns) == original_columns.union(
        {(segment, add_column) for segment in ts_non_negative.segments}
    )


@pytest.mark.parametrize(
    "inplace, segment, check_column, function, inverse_function, expected_result",
    [
        (False, "1", "target_transformed", lambda x: x**2, None, np.array([i**2 for i in range(100)])),
        (True, "1", "target", lambda x: x**2, lambda x: x**0.5, np.array([i**2 for i in range(100)])),
        (False, "2", "target_transformed", lambda x: x**2, None, np.array([1, 9] * 50)),
        (True, "2", "target", lambda x: x**2, lambda x: x**0.5, np.array([1, 9] * 50)),
    ],
)
def test_transform(ts_range_const, inplace, check_column, function, inverse_function, expected_result, segment):
    transform = LambdaTransform(
        in_column="target",
        transform_func=function,
        inplace=inplace,
        inverse_transform_func=inverse_function,
        out_column=check_column,
    )
    ts_range_const.fit_transform([transform])
    np.testing.assert_allclose(np.array(ts_range_const[:, segment, check_column]), expected_result, rtol=1e-9)


@pytest.mark.parametrize(
    "function, inverse_function",
    [(lambda x: x**2, lambda x: x**0.5)],
)
def test_inverse_transform(ts_range_const, function, inverse_function):
    transform = LambdaTransform(
        in_column="target", transform_func=function, inplace=True, inverse_transform_func=inverse_function
    )
    original_df = ts_range_const.to_pandas()
    ts_range_const.fit_transform([transform])
    ts_range_const.inverse_transform()
    check_column = "target"
    for segment in ts_range_const.segments:
        np.testing.assert_allclose(
            ts_range_const[:, segment, check_column], original_df[(segment, check_column)], rtol=1e-9
        )
