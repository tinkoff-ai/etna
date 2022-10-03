import numpy as np
import pandas as pd
import pytest

from etna.datasets import TSDataset
from etna.metrics import R2
from etna.models import LinearPerSegmentModel
from etna.transforms.timestamp import FourierTransform


def add_seasonality(series: pd.Series, period: int, magnitude: float) -> pd.Series:
    """Add seasonality to given series."""
    new_series = series.copy()
    size = series.shape[0]
    indices = np.arange(size)
    new_series += np.sin(2 * np.pi * indices / period) * magnitude
    return new_series


def get_one_df(period_1, period_2, magnitude_1, magnitude_2):
    timestamp = pd.date_range(start="2020-01-01", end="2021-01-01", freq="D")
    df = pd.DataFrame({"timestamp": timestamp})
    target = 0
    indices = np.arange(timestamp.shape[0])
    target += np.sin(2 * np.pi * indices * 2 / period_1) * magnitude_1
    target += np.cos(2 * np.pi * indices * 3 / period_2) * magnitude_2
    target += np.random.normal(scale=0.05, size=timestamp.shape[0])
    df["target"] = target
    return df


@pytest.fixture
def ts_trend_seasonal(random_seed) -> TSDataset:
    df_1 = get_one_df(period_1=7, period_2=30.4, magnitude_1=1, magnitude_2=1 / 2)
    df_1["segment"] = "segment_1"
    df_2 = get_one_df(period_1=7, period_2=30.4, magnitude_1=1 / 2, magnitude_2=1 / 5)
    df_2["segment"] = "segment_2"
    classic_df = pd.concat([df_1, df_2], ignore_index=True)
    return TSDataset(TSDataset.to_dataset(classic_df), freq="D")


@pytest.mark.parametrize("order, mods, repr_mods", [(None, [1, 2, 3, 4], [1, 2, 3, 4]), (2, None, [1, 2, 3, 4])])
def test_repr(order, mods, repr_mods):
    transform = FourierTransform(
        period=10,
        order=order,
        mods=mods,
    )
    transform_repr = transform.__repr__()
    true_repr = f"FourierTransform(period = 10, order = None, mods = {repr_mods}, out_column = None, )"
    assert transform_repr == true_repr


@pytest.mark.parametrize("period", [-1, 0, 1, 1.5])
def test_fail_period(period):
    """Test that transform is not created with wrong period."""
    with pytest.raises(ValueError, match="Period should be at least 2"):
        _ = FourierTransform(period=period, order=1)


@pytest.mark.parametrize("order", [0, 5])
def test_fail_order(order):
    """Test that transform is not created with wrong order."""
    with pytest.raises(ValueError, match="Order should be within"):
        _ = FourierTransform(period=7, order=order)


@pytest.mark.parametrize("mods", [[0], [0, 1, 2, 3], [1, 2, 3, 7], [7]])
def test_fail_mods(mods):
    """Test that transform is not created with wrong mods."""
    with pytest.raises(ValueError, match="Every mod should be within"):
        _ = FourierTransform(period=7, mods=mods)


def test_fail_set_none():
    """Test that transform is not created without order and mods."""
    with pytest.raises(ValueError, match="There should be exactly one option set"):
        _ = FourierTransform(period=7)


def test_fail_set_both():
    """Test that transform is not created with both order and mods set."""
    with pytest.raises(ValueError, match="There should be exactly one option set"):
        _ = FourierTransform(period=7, order=1, mods=[1, 2, 3])


@pytest.mark.parametrize(
    "period, order, num_columns", [(6, 2, 4), (7, 2, 4), (6, 3, 5), (7, 3, 6), (5.5, 2, 4), (5.5, 3, 5)]
)
def test_column_names(example_df, period, order, num_columns):
    """Test that transform creates expected number of columns and they can be recreated by its name."""
    df = TSDataset.to_dataset(example_df)
    segments = df.columns.get_level_values("segment").unique()
    transform = FourierTransform(period=period, order=order)
    transformed_df = transform.fit_transform(df)
    columns = transformed_df.columns.get_level_values("feature").unique().drop("target")
    assert len(columns) == num_columns
    for column in columns:
        transform_temp = eval(column)
        df_temp = transform_temp.fit_transform(df)
        columns_temp = df_temp.columns.get_level_values("feature").unique().drop("target")
        assert len(columns_temp) == 1
        generated_column = columns_temp[0]
        assert generated_column == column
        assert np.all(
            df_temp.loc[:, pd.IndexSlice[segments, generated_column]]
            == transformed_df.loc[:, pd.IndexSlice[segments, column]]
        )


def test_column_names_out_column(example_df):
    """Test that transform creates expected columns if `out_column` is set"""
    df = TSDataset.to_dataset(example_df)
    transform = FourierTransform(period=10, order=3, out_column="regressor_fourier")
    transformed_df = transform.fit_transform(df)
    columns = transformed_df.columns.get_level_values("feature").unique().drop("target")
    expected_columns = {f"regressor_fourier_{i}" for i in range(1, 7)}
    assert set(columns) == expected_columns


@pytest.mark.parametrize("period, mod", [(24, 1), (24, 2), (24, 9), (24, 20), (24, 23), (7.5, 3), (7.5, 4)])
def test_column_values(example_df, period, mod):
    """Test that transform generates correct values."""
    df = TSDataset.to_dataset(example_df)
    transform = FourierTransform(period=period, mods=[mod], out_column="regressor_fourier")
    transformed_df = transform.fit_transform(df)
    for segment in example_df["segment"].unique():
        transform_values = transformed_df.loc[:, pd.IndexSlice[segment, f"regressor_fourier_{mod}"]]

        timestamp = df.index
        freq = pd.Timedelta("1H")
        elapsed = (timestamp - timestamp[0]) / (period * freq)
        order = (mod + 1) // 2
        if mod % 2 == 0:
            expected_values = np.cos(2 * np.pi * order * elapsed).values
        else:
            expected_values = np.sin(2 * np.pi * order * elapsed).values

        assert np.allclose(transform_values, expected_values, atol=1e-12)


def test_forecast(ts_trend_seasonal):
    """Test that transform works correctly in forecast."""
    transform_1 = FourierTransform(period=7, order=3)
    transform_2 = FourierTransform(period=30.4, order=5)
    ts_train, ts_test = ts_trend_seasonal.train_test_split(test_size=10)
    ts_train.fit_transform(transforms=[transform_1, transform_2])
    model = LinearPerSegmentModel()
    model.fit(ts_train)
    ts_future = ts_train.make_future(10)
    ts_forecast = model.forecast(ts_future)
    metric = R2("macro")
    r2 = metric(ts_test, ts_forecast)
    assert r2 > 0.95
