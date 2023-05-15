from typing import List
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
import pytest
from typing_extensions import assert_never

from etna.datasets import TSDataset
from etna.metrics import R2
from etna.models import LinearPerSegmentModel
from etna.models import ProphetModel
from etna.pipeline import Pipeline
from etna.transforms import LagTransform
from etna.transforms.math import DifferencingTransform
from etna.transforms.math.differencing import _SingleDifferencingTransform
from tests.test_transforms.utils import assert_sampling_is_valid
from tests.test_transforms.utils import assert_transformation_equals_loaded_original
from tests.utils import select_segments_subset

GeneralDifferencingTransform = Union[_SingleDifferencingTransform, DifferencingTransform]


def extract_new_features_columns(transformed_df: pd.DataFrame, initial_df: pd.DataFrame) -> List[str]:
    """Extract columns from feature level that are present in transformed_df but not in initial_df."""
    return (
        transformed_df.columns.get_level_values("feature")
        .difference(initial_df.columns.get_level_values("feature"))
        .unique()
        .tolist()
    )


@pytest.fixture
def df_nans() -> pd.DataFrame:
    """Create DataFrame with nans at the beginning of one segment."""
    timestamp = pd.date_range("2021-01-01", "2021-04-01")
    df_1 = pd.DataFrame({"timestamp": timestamp, "target": np.arange(timestamp.shape[0]), "segment": "1"})
    df_2 = pd.DataFrame({"timestamp": timestamp[5:], "target": np.arange(timestamp[5:].shape[0]) * 2, "segment": "2"})
    df = pd.concat([df_1, df_2], ignore_index=True)
    df = TSDataset.to_dataset(df)
    return df


@pytest.fixture
def df_regressors() -> pd.DataFrame:
    """Create df_exog for df_nans."""
    timestamp = pd.date_range("2021-01-01", "2021-05-01")
    df_1 = pd.DataFrame({"timestamp": timestamp, "regressor_1": np.sin(np.arange(timestamp.shape[0])), "segment": "1"})
    df_2 = pd.DataFrame(
        {"timestamp": timestamp[5:], "regressor_1": np.sin(np.arange(timestamp[5:].shape[0])) * 2, "segment": "2"}
    )
    df = pd.concat([df_1, df_2], ignore_index=True)
    df = TSDataset.to_dataset(df)
    return df


@pytest.fixture
def ts_nans(df_nans) -> TSDataset:
    """Create TSDataset with nans at the beginning of one segment."""
    ts = TSDataset(df=df_nans, freq="D")
    return ts


@pytest.fixture
def ts_nans_with_regressors(df_nans, df_regressors) -> TSDataset:
    """Create TSDataset with regressors and nans at the beginning of one segment."""
    ts = TSDataset(df=df_nans, df_exog=df_regressors, freq="D")
    return ts


@pytest.fixture
def ts_nans_middle() -> TSDataset:
    """Create TSDataset with nans in the middle of the segment."""
    timestamp = pd.date_range("2021-01-01", "2021-04-01")
    df_1 = pd.DataFrame({"timestamp": timestamp, "target": np.arange(timestamp.shape[0]), "segment": "1"})
    df_2 = pd.DataFrame({"timestamp": timestamp, "target": np.arange(timestamp.shape[0]) * 2, "segment": "2"})
    df = pd.concat([df_1, df_2], ignore_index=True)
    df = TSDataset.to_dataset(df)
    df.iloc[5:10, 0] = np.NaN
    ts = TSDataset(df=df, freq="D")
    return ts


@pytest.fixture
def ts_segments_split(ts_nans) -> Tuple[TSDataset, TSDataset]:
    """Create a pair of datasets with different segments."""
    train_ts = select_segments_subset(ts=ts_nans, segments=["1"])
    test_ts = select_segments_subset(ts=ts_nans, segments=["2"])
    return train_ts, test_ts


@pytest.fixture
def ts_nans_with_noise(df_nans, random_seed) -> TSDataset:
    """Create noised version of df_nans."""
    df_nans.loc[:, pd.IndexSlice["1", "target"]] += np.random.normal(scale=0.03, size=df_nans.shape[0])
    df_nans.loc[df_nans.index[5] :, pd.IndexSlice["2", "target"]] += np.random.normal(
        scale=0.05, size=df_nans.shape[0] - 5
    )
    ts = TSDataset(df=df_nans, freq="D")
    return ts


def check_interface_transform_autogenerate_column_non_regressor(transform: GeneralDifferencingTransform, ts: TSDataset):
    """Check that differencing transform generates non-regressor column in transform according to repr."""
    df = ts.to_pandas()
    transformed_df = transform.fit_transform(ts).to_pandas()
    new_columns = set(extract_new_features_columns(transformed_df, df))
    assert new_columns == {repr(transform)}


def check_interface_transform_autogenerate_column_regressor(transform: GeneralDifferencingTransform, ts: TSDataset):
    """Check that differencing transform generates regressor column in transform according to repr."""
    df = ts.to_pandas()
    transformed_df = transform.fit_transform(ts).to_pandas()
    new_columns = set(extract_new_features_columns(transformed_df, df))
    assert new_columns == {repr(transform)}


def check_transform(
    transform: GeneralDifferencingTransform,
    period: int,
    order: int,
    out_column: str,
    fit_ts: TSDataset,
    ts: TSDataset,
):
    """Check that differencing transform generates correct values in transform."""
    df = ts.to_pandas()
    transform.fit(fit_ts)
    transformed_df = transform.transform(ts).to_pandas()

    for segment in df.columns.get_level_values("segment").unique():
        series_init = df.loc[:, pd.IndexSlice[segment, "target"]]
        series_transformed = transformed_df.loc[:, pd.IndexSlice[segment, out_column]]

        for _ in range(order):
            series_init = series_init.diff(periods=period)

        assert series_init.equals(series_transformed)


def check_inverse_transform_not_inplace(
    transform: GeneralDifferencingTransform, train_ts: TSDataset, test_ts: TSDataset
):
    """Check that differencing transform does nothing during inverse_transform in non-inplace mode."""
    transform.fit_transform(train_ts)
    transformed_df = transform.transform(test_ts).to_pandas()
    inverse_transformed_df = transform.inverse_transform(test_ts).to_pandas()
    pd.testing.assert_frame_equal(transformed_df, inverse_transformed_df)


def check_inverse_transform_inplace_train(transform: GeneralDifferencingTransform, ts: TSDataset):
    """Check that differencing transform correctly makes inverse_transform on train data in inplace mode."""
    df = ts.to_pandas()
    transform.fit_transform(ts)
    inverse_transformed_df = transform.inverse_transform(ts).to_pandas()
    pd.testing.assert_frame_equal(inverse_transformed_df, df)


def check_inverse_transform_inplace_filled_test(
    transform: GeneralDifferencingTransform, period: int, order: int, ts: TSDataset
):
    """Check that differencing transform correctly makes inverse_transform on filled test data in inplace mode."""
    ts_train, ts_test = ts.train_test_split(test_size=20)
    transform.fit(ts_train)

    # make predictions by hand taking into account the nature of ts_nans
    future_ts = ts_train.make_future(20, transforms=[transform])
    if order == 1:
        future_ts.df.loc[:, pd.IndexSlice["1", "target"]] = 1 * period
        future_ts.df.loc[:, pd.IndexSlice["2", "target"]] = 2 * period
    elif order >= 2:
        future_ts.df.loc[:, pd.IndexSlice["1", "target"]] = 0
        future_ts.df.loc[:, pd.IndexSlice["2", "target"]] = 0
    else:
        assert_never(order)

    # check values from inverse_transform
    future_ts.inverse_transform([transform])
    assert np.all(future_ts.to_pandas() == ts_test.to_pandas())


def check_inverse_transform_inplace_unfilled_test(transform: GeneralDifferencingTransform, ts: TSDataset):
    """Check that differencing transform correctly makes inverse_transform on unfilled test data in inplace mode."""
    ts_train, ts_test = ts.train_test_split(test_size=20)
    ts_train.fit_transform(transforms=[transform])

    future_ts = ts_train.make_future(20)

    # check values from inverse_transform
    future_ts.inverse_transform([transform])
    assert future_ts.to_pandas().isna().all().all()


def check_inverse_transform_inplace_test_quantiles(transform: GeneralDifferencingTransform, ts: TSDataset):
    """Check that differencing transform correctly makes inverse_transform on test data with quantiles."""
    ts_train, ts_test = ts.train_test_split(test_size=20)
    ts_train.fit_transform(transforms=[transform])
    model = ProphetModel()
    model.fit(ts_train)

    # make predictions by Prophet with prediction interval
    future_ts = ts_train.make_future(20, transforms=[transform])
    predict_ts = model.forecast(future_ts, prediction_interval=True, quantiles=[0.025, 0.975])
    predict_ts.inverse_transform([transform])

    # check that predicted value is within the interval
    for segment in predict_ts.segments:
        assert np.all(predict_ts[:, segment, "target_0.025"] <= predict_ts[:, segment, "target"])
        assert np.all(predict_ts[:, segment, "target"] <= predict_ts[:, segment, "target_0.975"])


def check_backtest_sanity(transform: GeneralDifferencingTransform, ts: TSDataset):
    """Check that differencing transform correctly works in backtest."""
    # create pipeline with linear model
    model = LinearPerSegmentModel()
    pipeline = Pipeline(
        model=model, transforms=[LagTransform(in_column="target", lags=[7, 8, 9]), transform], horizon=7
    )

    # run backtest
    metrics_df, _, _ = pipeline.backtest(ts, n_folds=3, aggregate_metrics=True, metrics=[R2()])
    assert np.all(metrics_df["R2"] > 0.95)


def test_single_fail_wrong_period():
    """Test that _SingleDifferencingTransform can't be created with period < 1."""
    with pytest.raises(ValueError, match="Period should be at least 1"):
        _ = _SingleDifferencingTransform(in_column="target", period=0, inplace=False, out_column="diff")


def test_full_fail_wrong_period():
    """Test that DifferencingTransform can't be created with period < 1."""
    with pytest.raises(ValueError, match="Period should be at least 1"):
        _ = DifferencingTransform(in_column="target", period=0, inplace=False, out_column="diff")


def test_full_fail_wrong_order():
    """Test that DifferencingTransform can't be created with order < 1."""
    with pytest.raises(ValueError, match="Order should be at least 1"):
        _ = DifferencingTransform(in_column="target", period=1, order=0, inplace=False, out_column="diff")


@pytest.mark.parametrize(
    "transform",
    [
        _SingleDifferencingTransform(in_column="target", period=1, inplace=False, out_column="diff"),
        DifferencingTransform(in_column="target", period=1, inplace=False, out_column="diff"),
    ],
)
def test_general_interface_transform_out_column(transform, ts_nans):
    """Test that differencing transform generates new column in transform according to out_column parameter."""
    df = ts_nans.to_pandas()
    transformed_df = transform.fit_transform(ts_nans).to_pandas()
    new_columns = set(extract_new_features_columns(transformed_df, df))
    assert new_columns == {"diff"}


@pytest.mark.parametrize("period", [1, 7])
def test_single_interface_transform_autogenerate_column_non_regressor(period, ts_nans):
    """Test that _SingleDifferencingTransform generates non-regressor column in transform according to repr."""
    transform = _SingleDifferencingTransform(in_column="target", period=period, inplace=False)
    check_interface_transform_autogenerate_column_non_regressor(transform, ts_nans)


@pytest.mark.parametrize("period", [1, 7])
@pytest.mark.parametrize("order", [1, 2])
def test_full_interface_transform_autogenerate_column_non_regressor(period, order, ts_nans):
    """Test that DifferencingTransform generates non-regressor column in transform according to repr."""
    transform = DifferencingTransform(in_column="target", period=period, order=order, inplace=False)
    check_interface_transform_autogenerate_column_non_regressor(transform, ts_nans)


@pytest.mark.parametrize("period", [1, 7])
def test_single_interface_transform_autogenerate_column_regressor(period, ts_nans_with_regressors):
    """Test that _SingleDifferencingTransform generates regressor column in transform according to repr."""
    transform = _SingleDifferencingTransform(in_column="regressor_1", period=period, inplace=False)
    check_interface_transform_autogenerate_column_regressor(transform, ts_nans_with_regressors)


@pytest.mark.parametrize("period", [1, 7])
@pytest.mark.parametrize("order", [1, 2])
def test_full_interface_transform_autogenerate_column_regressor(period, order, ts_nans_with_regressors):
    """Test that DifferencingTransform generates regressor column in transform according to repr."""
    transform = DifferencingTransform(in_column="regressor_1", period=period, order=order, inplace=False)
    check_interface_transform_autogenerate_column_regressor(transform, ts_nans_with_regressors)


@pytest.mark.parametrize(
    "transform",
    [
        _SingleDifferencingTransform(in_column="target", period=1, inplace=True),
        DifferencingTransform(in_column="target", period=1, order=1, inplace=True),
    ],
)
def test_general_interface_transform_inplace(transform, ts_nans):
    """Test that differencing transform doesn't generate new column in transform in inplace mode."""
    df = ts_nans.to_pandas()
    transform = _SingleDifferencingTransform(in_column="target", period=1, inplace=True)
    transformed_df = transform.fit_transform(ts_nans).to_pandas()

    new_columns = set(extract_new_features_columns(transformed_df, df))
    assert len(new_columns) == 0


@pytest.mark.parametrize(
    "transform",
    [
        _SingleDifferencingTransform(in_column="target", period=1, inplace=False, out_column="diff"),
        DifferencingTransform(in_column="target", period=1, order=1, inplace=False, out_column="diff"),
    ],
)
def test_general_interface_transform_not_inplace(transform, ts_nans):
    """Test that differencing transform doesn't change in_column in transform in non-inplace mode."""
    df = ts_nans.to_pandas()
    transformed_df = transform.fit_transform(ts_nans).to_pandas()

    transformed_df_compare = transformed_df[df.columns]
    pd.testing.assert_frame_equal(df, transformed_df_compare)


@pytest.mark.parametrize(
    "transform",
    [
        _SingleDifferencingTransform(in_column="target", period=1, inplace=False, out_column="diff"),
        DifferencingTransform(in_column="target", period=1, order=1, inplace=False, out_column="diff"),
    ],
)
def test_general_fit_fail_nans(transform, ts_nans):
    """Test that differencing transform fails to fit on segments with NaNs inside."""
    # put nans inside one segment
    ts_nans.df.iloc[-3, 0] = np.NaN

    with pytest.raises(ValueError, match="There should be no NaNs inside the segments"):
        transform.fit(ts_nans)


@pytest.mark.parametrize("inplace, out_column", [(False, "diff"), (True, "target")])
def test_full_transform_fail_not_fitted(inplace, out_column, ts_nans):
    """Test that DifferencingTransform transform fails to make transform before fitting."""
    transform = DifferencingTransform(in_column="target", inplace=inplace, out_column=out_column)
    with pytest.raises(ValueError, match="Transform is not fitted"):
        _ = transform.transform(ts_nans)


@pytest.mark.parametrize("period", [1, 7])
def test_single_transform_inplace_new_segments(period, ts_segments_split):
    """Test that _SingleDifferencingTransform generates correct values in transform on new segments in inplace mode."""
    train_ts, test_ts = ts_segments_split
    transform = _SingleDifferencingTransform(in_column="target", period=period, inplace=True)
    check_transform(transform, period, 1, "target", train_ts, test_ts)


def test_full_transform_inplace_fail_new_segments(ts_segments_split):
    """Test that DifferencingTransform transform fails to make transform if new segments are present in inplace mode."""
    train_ts, test_ts = ts_segments_split
    transform = DifferencingTransform(in_column="target", period=1, order=1, inplace=True)
    transform.fit(train_ts)
    with pytest.raises(
        NotImplementedError, match="This transform can't process segments that weren't present on train data"
    ):
        _ = transform.transform(test_ts)


@pytest.mark.parametrize("period", [1, 7])
@pytest.mark.parametrize("inplace, out_column", [(False, "diff"), (True, "target")])
def test_single_transform(period, inplace, out_column, ts_nans):
    """Test that _SingleDifferencingTransform generates correct values in transform."""
    transform = _SingleDifferencingTransform(in_column="target", period=period, inplace=inplace, out_column=out_column)
    check_transform(transform, period, 1, out_column, ts_nans, ts_nans)


@pytest.mark.parametrize("period", [1, 7])
@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("inplace, out_column", [(False, "diff"), (True, "target")])
def test_full_transform(period, order, inplace, out_column, ts_nans):
    """Test that DifferencingTransform generates correct values in transform."""
    transform = DifferencingTransform(
        in_column="target", period=period, order=order, inplace=inplace, out_column=out_column
    )
    check_transform(transform, period, order, out_column, ts_nans, ts_nans)


@pytest.mark.parametrize("period", [1, 7])
@pytest.mark.parametrize("inplace, out_column", [(False, "diff"), (True, "target")])
def test_single_transform_nans_middle(period, inplace, out_column, ts_nans, ts_nans_middle):
    """Test that _SingleDifferencingTransform generates correct values in transform with NaNs in the middle."""
    transform = _SingleDifferencingTransform(in_column="target", period=period, inplace=inplace, out_column=out_column)
    check_transform(transform, period, 1, out_column, ts_nans, ts_nans_middle)


@pytest.mark.parametrize("period", [1, 7])
@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("inplace, out_column", [(False, "diff"), (True, "target")])
def test_full_transform_nans_middle(period, order, inplace, out_column, ts_nans, ts_nans_middle):
    """Test that DifferencingTransform generates correct values in transform with NaNs in the middle."""
    transform = DifferencingTransform(
        in_column="target", period=period, order=order, inplace=inplace, out_column=out_column
    )
    check_transform(transform, period, order, out_column, ts_nans, ts_nans_middle)


@pytest.mark.parametrize("period", [1, 7])
def test_single_transform_not_inplace_new_segments(period, ts_segments_split):
    """Test that _SingleDifferencingTransform generates correct values in transform on new segments in non-inplace mode."""
    train_ts, test_ts = ts_segments_split
    out_column = "diff"
    transform = _SingleDifferencingTransform(in_column="target", period=period, inplace=False, out_column=out_column)
    check_transform(transform, period, 1, out_column, train_ts, test_ts)


@pytest.mark.parametrize("period", [1, 7])
@pytest.mark.parametrize("order", [1, 2])
def test_full_transform_not_inplace_new_segments(period, order, ts_segments_split):
    """Test that DifferencingTransform generates correct values in transform on new segments in non-inplace mode."""
    train_ts, test_ts = ts_segments_split
    out_column = "diff"
    transform = DifferencingTransform(
        in_column="target", period=period, order=order, inplace=False, out_column=out_column
    )
    check_transform(transform, period, order, out_column, train_ts, test_ts)


@pytest.mark.parametrize("inplace, out_column", [(False, "diff"), (True, "target")])
def test_full_inverse_transform_fail_not_fitted(inplace, out_column, ts_nans):
    """Test that DifferencingTransform fails to make inverse_transform before fitting."""
    transform = DifferencingTransform(in_column="target", inplace=inplace, out_column=out_column)
    with pytest.raises(ValueError, match="Transform is not fitted"):
        _ = transform.inverse_transform(ts_nans)


def test_full_inverse_transform_inplace_fail_new_segments(ts_segments_split):
    """Test that DifferencingTransform fails to make inverse_transform if new segments are present in inplace mode."""
    train_ts, test_ts = ts_segments_split
    transform = DifferencingTransform(in_column="target", period=1, order=1, inplace=True)
    transform.fit(train_ts)
    with pytest.raises(
        NotImplementedError, match="This transform can't process segments that weren't present on train data"
    ):
        _ = transform.inverse_transform(test_ts)


@pytest.mark.parametrize(
    "transform",
    [
        _SingleDifferencingTransform(in_column="target", period=1, inplace=True),
        DifferencingTransform(in_column="target", period=1, order=1, inplace=True),
    ],
)
def test_general_inverse_transform_fail_not_all_test(transform, ts_nans):
    """Test that differencing transform fails to make inverse_transform only on part of train."""
    transform.fit_transform(ts_nans)
    ts_nans.df = ts_nans.df.iloc[1:]

    with pytest.raises(ValueError, match="Inverse transform can be applied only to full train"):
        transform.inverse_transform(ts_nans)


@pytest.mark.parametrize(
    "transform",
    [
        _SingleDifferencingTransform(in_column="target", period=1, inplace=True),
        DifferencingTransform(in_column="target", period=1, order=1, inplace=True),
    ],
)
def test_general_inverse_transform_fail_test_not_right_after_train(transform, ts_nans):
    """Test that differencing transform fails to make inverse_transform on not adjacent test data."""
    ts = ts_nans
    ts_train, ts_test = ts.train_test_split(test_size=10)
    ts_train.fit_transform(transforms=[transform])
    future_ts = ts_train.make_future(10, transforms=[transform])
    future_ts_cropped = TSDataset(future_ts.to_pandas().iloc[1:], freq=future_ts.freq)

    with pytest.raises(ValueError, match="Test should go after the train without gaps"):
        _ = transform.inverse_transform(future_ts_cropped)


@pytest.mark.parametrize("period", [1, 7])
def test_single_inverse_transform_not_inplace(period, ts_nans):
    """Test that _SingleDifferencingTransform does nothing during inverse_transform in non-inplace mode."""
    transform = _SingleDifferencingTransform(in_column="target", period=period, inplace=False, out_column="diff")
    check_inverse_transform_not_inplace(transform, ts_nans, ts_nans)


@pytest.mark.parametrize("period", [1, 7])
@pytest.mark.parametrize("order", [1, 2])
def test_full_inverse_transform_not_inplace(period, order, ts_nans):
    """Test that DifferencingTransform does nothing during inverse_transform in non-inplace mode."""
    transform = DifferencingTransform(in_column="target", period=period, order=order, inplace=False, out_column="diff")
    check_inverse_transform_not_inplace(transform, ts_nans, ts_nans)


@pytest.mark.parametrize("period", [1, 7])
def test_single_inverse_transform_not_inplace_new_segments(period, ts_segments_split):
    """Test that _SingleDifferencingTransform does nothing during inverse_transform on new segments in non-inplace mode."""
    train_ts, test_ts = ts_segments_split
    transform = _SingleDifferencingTransform(in_column="target", period=period, inplace=False, out_column="diff")
    check_inverse_transform_not_inplace(transform, train_ts, test_ts)


@pytest.mark.parametrize("period", [1, 7])
@pytest.mark.parametrize("order", [1, 2])
def test_full_inverse_transform_not_inplace_new_segments(period, order, ts_segments_split):
    """Test that DifferencingTransform does nothing during inverse_transform on new segments in non-inplace mode."""
    train_ts, test_ts = ts_segments_split
    transform = DifferencingTransform(in_column="target", period=period, order=order, inplace=False, out_column="diff")
    check_inverse_transform_not_inplace(transform, train_ts, test_ts)


@pytest.mark.parametrize("period", [1, 7])
def test_single_inverse_transform_inplace_train(period, ts_nans):
    """Test that _SingleDifferencingTransform correctly makes inverse_transform on train data in inplace mode."""
    transform = _SingleDifferencingTransform(in_column="target", period=period, inplace=True)
    check_inverse_transform_inplace_train(transform, ts_nans)


@pytest.mark.parametrize("period", [1, 7])
@pytest.mark.parametrize("order", [1, 2])
def test_full_inverse_transform_inplace_train(period, order, ts_nans):
    """Test that DifferencingTransform correctly makes inverse_transform on train data in inplace mode."""
    transform = DifferencingTransform(in_column="target", period=period, order=order, inplace=True)
    check_inverse_transform_inplace_train(transform, ts_nans)


@pytest.mark.parametrize("period", [1, 7])
def test_single_inverse_transform_inplace_filled_test(period, ts_nans):
    """Test that _SingleDifferencingTransform correctly makes inverse_transform on filled test data in inplace mode."""
    transform = _SingleDifferencingTransform(in_column="target", period=period, inplace=True)
    check_inverse_transform_inplace_filled_test(transform, period, 1, ts_nans)


@pytest.mark.parametrize("period", [1, 7])
@pytest.mark.parametrize("order", [1, 2])
def test_full_inverse_transform_inplace_test(period, order, ts_nans):
    """Test that DifferencingTransform correctly makes inverse_transform on filled test data in inplace mode."""
    transform = DifferencingTransform(in_column="target", period=period, order=order, inplace=True)
    check_inverse_transform_inplace_filled_test(transform, period, order, ts_nans)


@pytest.mark.parametrize("period", [1, 7])
def test_single_inverse_transform_inplace_test(period, ts_nans):
    """Test that _SingleDifferencingTransform correctly makes inverse_transform on unfilled test data in inplace mode."""
    transform = _SingleDifferencingTransform(in_column="target", period=period, inplace=True)
    check_inverse_transform_inplace_unfilled_test(transform, ts_nans)


@pytest.mark.parametrize("period", [1, 7])
@pytest.mark.parametrize("order", [1, 2])
def test_full_inverse_transform_inplace_test(period, order, ts_nans):
    """Test that DifferencingTransform correctly makes inverse_transform on unfilled test data in inplace mode."""
    transform = DifferencingTransform(in_column="target", period=period, order=order, inplace=True)
    check_inverse_transform_inplace_unfilled_test(transform, ts_nans)


@pytest.mark.parametrize("period", [1, 7])
def test_single_inverse_transform_inplace_test_quantiles(period, ts_nans_with_noise):
    """Test that _SingleDifferencingTransform correctly makes inverse_transform on test data with quantiles."""
    transform = _SingleDifferencingTransform(in_column="target", period=period, inplace=True)
    check_inverse_transform_inplace_test_quantiles(transform, ts_nans_with_noise)


@pytest.mark.parametrize("period", [1, 7])
@pytest.mark.parametrize("order", [1, 2])
def test_full_inverse_transform_inplace_test_quantiles(period, order, ts_nans_with_noise):
    """Test that DifferencingTransform correctly makes inverse_transform on test data with quantiles."""
    transform = DifferencingTransform(in_column="target", period=period, order=2, inplace=True)
    check_inverse_transform_inplace_test_quantiles(transform, ts_nans_with_noise)


@pytest.mark.parametrize("period", [1, 7])
def test_single_backtest_sanity(period, ts_nans_with_noise):
    """Test that _SingleDifferencingTransform correctly works in backtest."""
    transform = _SingleDifferencingTransform(in_column="target", period=period, inplace=True)
    check_backtest_sanity(transform, ts_nans_with_noise)


@pytest.mark.parametrize("period", [1, 7])
@pytest.mark.parametrize("order", [1, 2])
def test_full_backtest_sanity(period, order, ts_nans_with_noise):
    """Test that DifferencingTransform correctly works in backtest."""
    transform = DifferencingTransform(in_column="target", period=period, order=order, inplace=True)
    check_backtest_sanity(transform, ts_nans_with_noise)


@pytest.mark.parametrize("inplace", [False, True])
def test_save_load(inplace, ts_nans):
    ts = ts_nans
    transform = DifferencingTransform(in_column="target", inplace=inplace)
    assert_transformation_equals_loaded_original(transform=transform, ts=ts)


def test_get_regressors_info_not_fitted():
    transform = DifferencingTransform(in_column="target")
    with pytest.raises(ValueError, match="Fit the transform to get the correct regressors info!"):
        _ = transform.get_regressors_info()


@pytest.mark.parametrize("transform", [DifferencingTransform(in_column="target")])
def test_params_to_tune(transform, ts_nans):
    ts = ts_nans
    assert len(transform.params_to_tune()) > 0
    assert_sampling_is_valid(transform=transform, ts=ts)
