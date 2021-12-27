from typing import List

import numpy as np
import pandas as pd
import pytest

from etna.datasets import TSDataset
from etna.metrics import R2
from etna.models import NaiveModel
from etna.models import ProphetModel
from etna.pipeline import Pipeline
from etna.transforms.differencing import _SingleDifferencingTransform


def extract_new_features_columns(transformed_df: pd.DataFrame, initial_df: pd.DataFrame) -> List[str]:
    """Extract columns from feature level that are present in transformed_df but not present in initial_df."""
    return (
        transformed_df.columns.get_level_values("feature")
        .difference(initial_df.columns.get_level_values("feature"))
        .unique()
        .tolist()
    )


def equals_with_nans(first_df: pd.DataFrame, second_df: pd.DataFrame) -> bool:
    """Compare two dataframes with consideration NaN == NaN is true."""
    if first_df.shape != second_df.shape:
        return False
    compare_result = (first_df.isna() & second_df.isna()) | (first_df == second_df)
    return np.all(compare_result)


@pytest.fixture
def df_nans() -> pd.DataFrame:
    timestamp = pd.date_range("2021-01-01", "2021-03-01")
    df_1 = pd.DataFrame({"timestamp": timestamp, "target": np.arange(timestamp.shape[0]), "segment": "1"})
    df_2 = pd.DataFrame({"timestamp": timestamp[5:], "target": np.arange(timestamp[5:].shape[0]) * 2, "segment": "2"})
    df = pd.concat([df_1, df_2], ignore_index=True)
    df = TSDataset.to_dataset(df)
    return df


@pytest.fixture
def df_regressors(df_nans) -> pd.DataFrame:
    timestamp = pd.date_range("2021-01-01", "2021-04-01")
    df_1 = pd.DataFrame({"timestamp": timestamp, "regressor_1": np.sin(np.arange(timestamp.shape[0])), "segment": "1"})
    df_2 = pd.DataFrame(
        {"timestamp": timestamp[5:], "regressor_1": np.sin(np.arange(timestamp[5:].shape[0])) * 2, "segment": "2"}
    )
    df = pd.concat([df_1, df_2], ignore_index=True)
    df = TSDataset.to_dataset(df)
    return df


@pytest.fixture
def df_nans_with_noise(df_nans, random_seed) -> pd.DataFrame:
    df_nans.loc[:, pd.IndexSlice["1", "target"]] += np.random.normal(scale=0.05, size=df_nans.shape[0])
    df_nans.loc[df_nans.index[5] :, pd.IndexSlice["2", "target"]] += np.random.normal(
        scale=0.05, size=df_nans.shape[0] - 5
    )
    return df_nans


def test_single_interface_transform_out_column(df_nans):
    """Test that _SingleDifferencingTransform generates new column in transform according to out_column parameter."""
    transform = _SingleDifferencingTransform(in_column="target", period=1, inplace=False, out_column="diff")
    transformed_df = transform.fit_transform(df_nans)

    new_columns = set(extract_new_features_columns(transformed_df, df_nans))
    assert new_columns == {"diff"}


@pytest.mark.parametrize("period", [1, 7])
def test_single_interface_transform_autogenerate_column_non_regressor(df_nans, period):
    """Test that _SingleDifferencingTransform generates non-regressor column in transform according to repr."""
    transform = _SingleDifferencingTransform(in_column="target", period=period, inplace=False)
    transformed_df = transform.fit_transform(df_nans)

    new_columns = set(extract_new_features_columns(transformed_df, df_nans))
    assert new_columns == {transform.__repr__()}


@pytest.mark.parametrize("period", [1, 7])
def test_single_interface_transform_autogenerate_column_regressor(df_nans, df_regressors, period):
    """Test that _SingleDifferencingTransform generates regressor column in transform according to repr."""
    transform = _SingleDifferencingTransform(in_column="regressor_1", period=period, inplace=False)
    ts = TSDataset(df=df_nans, df_exog=df_regressors, freq="D")
    transformed_df = transform.fit_transform(ts.to_pandas())

    new_columns = set(extract_new_features_columns(transformed_df, ts.to_pandas()))
    assert new_columns == {f"regressor_{transform.__repr__()}"}


def test_single_interface_transform_inplace(df_nans):
    """Test that _SingleDifferencingTransform doesn't generate new column in transform in inplace mode."""
    transform = _SingleDifferencingTransform(in_column="target", period=1, inplace=True)
    transformed_df = transform.fit_transform(df_nans)

    new_columns = set(extract_new_features_columns(transformed_df, df_nans))
    assert len(new_columns) == 0


@pytest.mark.parametrize("period", [1, 7])
def test_single_fit_fail_nans(df_nans, period):
    """Test that _SingleDifferencingTransform fails to fit on segments with NaNs inside."""
    # put nans inside one segment
    df_nans.iloc[-3, 0] = np.NaN

    transform = _SingleDifferencingTransform(in_column="target", period=period, inplace=False, out_column="diff")

    with pytest.raises(ValueError, match="There should be no NaNs inside the segments"):
        transform.fit(df_nans)


def test_single_transform_not_inplace(df_nans):
    """Test that _SingleDifferencingTransform doesn't change in_column in transform in non-inplace mode."""
    transform = _SingleDifferencingTransform(in_column="target", period=1, inplace=False, out_column="diff")
    transformed_df = transform.fit_transform(df_nans)

    transformed_df_compare = transformed_df[df_nans.columns]
    assert equals_with_nans(df_nans, transformed_df_compare)


@pytest.mark.parametrize("period", [1, 7])
@pytest.mark.parametrize("inplace, out_column", [(False, "diff"), (True, "target")])
def test_single_transform_fail_not_fitted(period, inplace, out_column, df_nans):
    """Test that _SingleDifferencingTransform fails to make transform before fitting."""
    transform = _SingleDifferencingTransform(in_column="target", period=period, inplace=inplace, out_column=out_column)
    with pytest.raises(AttributeError, match="Transform is not fitted"):
        _ = transform.transform(df_nans)


@pytest.mark.parametrize("period", [1, 7])
@pytest.mark.parametrize("inplace, out_column", [(False, "diff"), (True, "target")])
def test_single_transform(period, inplace, out_column, df_nans):
    """Test that _SingleDifferencingTransform generates correct values in transform."""
    transform = _SingleDifferencingTransform(in_column="target", period=period, inplace=inplace, out_column=out_column)
    transformed_df = transform.fit_transform(df_nans)

    for segment in df_nans.columns.get_level_values("segment").unique():
        series_init = df_nans.loc[:, pd.IndexSlice[segment, "target"]]
        series_transformed = transformed_df.loc[:, pd.IndexSlice[segment, out_column]]

        series_init = series_init.loc[series_init.first_valid_index() :]
        series_transformed = series_transformed.loc[series_transformed.first_valid_index() :]

        assert series_init.shape[0] == series_transformed.shape[0] + period
        assert np.all(series_init.diff(periods=period).iloc[period:] == series_transformed)


@pytest.mark.parametrize("period", [1, 7])
@pytest.mark.parametrize("inplace, out_column", [(False, "diff"), (True, "target")])
def test_single_inverse_transform_fail_not_fitted(period, inplace, out_column, df_nans):
    """Test that _SingleDifferencingTransform fails to make inverse_transform before fitting."""
    transform = _SingleDifferencingTransform(in_column="target", period=period, inplace=inplace, out_column=out_column)
    with pytest.raises(AttributeError, match="Transform is not fitted"):
        _ = transform.inverse_transform(df_nans)


def test_single_inverse_transform_fail_not_all_test(df_nans):
    """Test that _SingleDifferencingTransform fails to make inverse_transform only on part of train."""
    transform = _SingleDifferencingTransform(in_column="target", period=1, inplace=True)
    transformed_df = transform.fit_transform(df_nans)

    with pytest.raises(ValueError, match="Inverse transform can be applied only to full train"):
        _ = transform.inverse_transform(transformed_df.iloc[1:])


def test_single_inverse_transform_fail_test_not_right_after_train(df_nans):
    """Test that _SingleDifferencingTransform fails to make inverse_transform on not adjacent test data."""
    ts = TSDataset(df_nans, freq="D")
    ts_train, ts_test = ts.train_test_split(test_size=10)

    transform = _SingleDifferencingTransform(in_column="target", period=1, inplace=True)
    ts_train.fit_transform(transforms=[transform])

    future_ts = ts_train.make_future(10)
    future_df = future_ts.to_pandas()

    with pytest.raises(ValueError, match="Test should go after the train without gaps"):
        _ = transform.inverse_transform(future_df.iloc[1:])


def test_single_inverse_transform_not_inplace(df_nans):
    """Test that _SingleDifferencingTransform does nothing during inverse_transform in non-inplace mode."""
    transform = _SingleDifferencingTransform(in_column="target", period=1, inplace=False, out_column="diff")
    transformed_df = transform.fit_transform(df_nans)
    inverse_transformed_df = transform.inverse_transform(transformed_df)

    assert equals_with_nans(transformed_df, inverse_transformed_df)


def test_single_inverse_transform_inplace_train(df_nans):
    """Test that _SingleDifferencingTransform correctly makes inverse_transform on train data in inplace mode."""
    transform = _SingleDifferencingTransform(in_column="target", period=1, inplace=True)
    transformed_df = transform.fit_transform(df_nans)
    inverse_transformed_df = transform.inverse_transform(transformed_df)

    assert equals_with_nans(inverse_transformed_df, df_nans)


@pytest.mark.parametrize("period", [1, 7])
def test_single_inverse_transform_inplace_test_fail_nans(period, df_nans):
    """Test that _SingleDifferencingTransform fails to make inverse_transform on test data if there are NaNs."""
    ts = TSDataset(df_nans, freq="D")
    ts_train, ts_test = ts.train_test_split(test_size=20)

    transform = _SingleDifferencingTransform(in_column="target", period=period, inplace=True)
    ts_train.fit_transform(transforms=[transform])

    # make predictions by hand
    future_ts = ts_train.make_future(20)
    future_ts.df.loc[:, pd.IndexSlice["1", "target"]] = np.NaN
    future_ts.df.loc[:, pd.IndexSlice["2", "target"]] = 2 * period

    # check fail on inverse_transform
    with pytest.raises(ValueError, match="There should be no NaNs inside the segments"):
        future_ts.inverse_transform()


@pytest.mark.parametrize("period", [1, 7])
def test_single_inverse_transform_inplace_test(period, df_nans):
    """Test that _SingleDifferencingTransform correctly makes inverse_transform on test data in inplace mode."""
    ts = TSDataset(df_nans, freq="D")
    ts_train, ts_test = ts.train_test_split(test_size=20)

    transform = _SingleDifferencingTransform(in_column="target", period=period, inplace=True)
    ts_train.fit_transform(transforms=[transform])

    # make predictions by hand
    future_ts = ts_train.make_future(20)
    future_ts.df.loc[:, pd.IndexSlice["1", "target"]] = 1 * period
    future_ts.df.loc[:, pd.IndexSlice["2", "target"]] = 2 * period

    # check values from inverse_transform
    future_ts.inverse_transform()
    assert np.all(future_ts.to_pandas() == ts_test.to_pandas())


@pytest.mark.parametrize("period", [1, 7])
def test_single_inverse_transform_inplace_test_quantiles(period, df_nans_with_noise):
    """Test that _SingleDifferencingTransform correctly makes inverse_transform on test data with quantiles."""
    ts = TSDataset(df_nans_with_noise, freq="D")
    ts_train, ts_test = ts.train_test_split(test_size=20)

    transform = _SingleDifferencingTransform(in_column="target", period=period, inplace=True)
    ts_train.fit_transform(transforms=[transform])
    model = ProphetModel()
    model.fit(ts_train)

    # make predictions by Prophet with prediction interval
    future_ts = ts_train.make_future(20)
    predict_ts = model.forecast(future_ts, prediction_interval=True, quantiles=[0.025, 0.975])

    # check that predicted value is within the interval
    for segment in predict_ts.segments:
        assert np.all(predict_ts[:, segment, "target_0.025"] <= predict_ts[:, segment, "target"])
        assert np.all(predict_ts[:, segment, "target"] <= predict_ts[:, segment, "target_0.975"])


@pytest.mark.parametrize("period", [1, 7])
def test_single_backtest_sanity(period, df_nans_with_noise):
    """Test that _SingleDifferencingTransform correctly works in backtest."""
    transform = _SingleDifferencingTransform(in_column="target", period=period, inplace=True)

    # create pipeline with naive model
    ts = TSDataset(df_nans_with_noise, freq="D")
    model = NaiveModel(lag=period)
    pipeline = Pipeline(model=model, transforms=[transform], horizon=7)

    # run backtest
    metrics_df, _, _ = pipeline.backtest(ts, n_folds=3, aggregate_metrics=True, metrics=[R2()])
    assert np.all(metrics_df["R2"] > 0.95)
