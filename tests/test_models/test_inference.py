import numpy as np
import pandas as pd
import pytest
from pandas.util.testing import assert_frame_equal
from pytorch_forecasting.data import GroupNormalizer

from etna.datasets import TSDataset
from etna.models import AutoARIMAModel
from etna.models import BATSModel
from etna.models import CatBoostModelMultiSegment
from etna.models import CatBoostModelPerSegment
from etna.models import ElasticMultiSegmentModel
from etna.models import ElasticPerSegmentModel
from etna.models import HoltModel
from etna.models import HoltWintersModel
from etna.models import LinearMultiSegmentModel
from etna.models import LinearPerSegmentModel
from etna.models import MovingAverageModel
from etna.models import NaiveModel
from etna.models import ProphetModel
from etna.models import SARIMAXModel
from etna.models import SeasonalMovingAverageModel
from etna.models import SimpleExpSmoothingModel
from etna.models import TBATSModel
from etna.models.nn import DeepARModel
from etna.models.nn import TFTModel
from etna.transforms import LagTransform
from etna.transforms import PytorchForecastingTransform


def _test_forecast_in_sample_full(ts, model, transforms):
    df = ts.to_pandas()

    # fitting
    ts.fit_transform(transforms)
    model.fit(ts)

    # forecasting
    forecast_ts = TSDataset(df, freq="D")
    forecast_ts.transform(ts.transforms)
    forecast_ts.df.loc[:, pd.IndexSlice[:, "target"]] = np.NaN
    model.forecast(forecast_ts)

    # checking
    forecast_df = forecast_ts.to_pandas(flatten=True)
    assert not np.any(forecast_df["target"].isna())


def _test_forecast_in_sample_suffix(ts, model, transforms):
    df = ts.to_pandas()

    # fitting
    ts.fit_transform(transforms)
    model.fit(ts)

    # forecasting
    forecast_ts = TSDataset(df, freq="D")
    forecast_ts.transform(ts.transforms)
    forecast_ts.df.loc[:, pd.IndexSlice[:, "target"]] = np.NaN
    forecast_ts.df = forecast_ts.df.iloc[6:]
    model.forecast(forecast_ts)

    # checking
    forecast_df = forecast_ts.to_pandas(flatten=True)
    assert not np.any(forecast_df["target"].isna())


def _test_forecast_out_sample_prefix(ts, model, transforms):
    # fitting
    ts.fit_transform(transforms)
    model.fit(ts)
    # forecasting full
    forecast_full_ts = ts.make_future(5)

    import torch  # TODO: remove after fix at issue-802

    torch.manual_seed(11)

    model.forecast(forecast_full_ts)

    # forecasting only prefix
    forecast_prefix_ts = ts.make_future(5)
    forecast_prefix_ts.df = forecast_prefix_ts.df.iloc[:-2]

    torch.manual_seed(11)  # TODO: remove after fix at issue-802
    model.forecast(forecast_prefix_ts)

    # checking
    forecast_full_df = forecast_full_ts.to_pandas()
    forecast_prefix_df = forecast_prefix_ts.to_pandas()
    assert_frame_equal(forecast_prefix_df, forecast_full_df.iloc[:-2])


def _test_forecast_out_sample_suffix(ts, model, transforms):
    # fitting
    ts.fit_transform(transforms)
    model.fit(ts)

    # forecasting full
    forecast_full_ts = ts.make_future(5)
    model.forecast(forecast_full_ts)

    # forecasting only suffix
    forecast_gap_ts = ts.make_future(5)
    forecast_gap_ts.df = forecast_gap_ts.df.iloc[2:]
    model.forecast(forecast_gap_ts)

    # checking
    forecast_full_df = forecast_full_ts.to_pandas()
    forecast_gap_df = forecast_gap_ts.to_pandas()
    assert_frame_equal(forecast_gap_df, forecast_full_df.iloc[2:])


def _test_forecast_mixed_in_out_sample(ts, model, transforms):
    # fitting
    df = ts.to_pandas()
    ts.fit_transform(transforms)
    model.fit(ts)

    # forecasting mixed in-sample and out-sample
    future_ts = ts.make_future(5)
    future_df = future_ts.to_pandas().loc[:, pd.IndexSlice[:, "target"]]
    df_full = pd.concat((df, future_df))
    forecast_full_ts = TSDataset(df=df_full, freq=future_ts.freq)
    forecast_full_ts.transform(ts.transforms)
    forecast_full_ts.df.loc[:, pd.IndexSlice[:, "target"]] = np.NaN
    forecast_full_ts.df = forecast_full_ts.df.iloc[6:]
    model.forecast(forecast_full_ts)

    # forecasting only in sample
    forecast_in_sample_ts = TSDataset(df, freq="D")
    forecast_in_sample_ts.transform(ts.transforms)
    forecast_in_sample_ts.df.loc[:, pd.IndexSlice[:, "target"]] = np.NaN
    forecast_in_sample_ts.df = forecast_in_sample_ts.df.iloc[6:]
    model.forecast(forecast_in_sample_ts)

    # forecasting only out sample
    forecast_out_sample_ts = ts.make_future(5)
    model.forecast(forecast_out_sample_ts)

    # checking
    forecast_full_df = forecast_full_ts.to_pandas()
    forecast_in_sample_df = forecast_in_sample_ts.to_pandas()
    forecast_out_sample_df = forecast_out_sample_ts.to_pandas()
    assert_frame_equal(forecast_in_sample_df, forecast_full_df.iloc[:-5])
    assert_frame_equal(forecast_out_sample_df, forecast_full_df.iloc[-5:])


@pytest.mark.parametrize(
    "model, transforms",
    [
        (CatBoostModelMultiSegment(), [LagTransform(in_column="target", lags=[2, 3])]),
        (ProphetModel(), []),
        (SARIMAXModel(), []),
        (AutoARIMAModel(), []),
        (HoltModel(), []),
        (HoltWintersModel(), []),
        (SimpleExpSmoothingModel(), []),
        (MovingAverageModel(window=3), []),
        (NaiveModel(lag=3), []),
        (SeasonalMovingAverageModel(), []),
    ],
)
def test_forecast_in_sample_full(model, transforms, example_tsds):
    _test_forecast_in_sample_full(example_tsds, model, transforms)


@pytest.mark.xfail(strict=True)
@pytest.mark.parametrize(
    "model, transforms",
    [
        (CatBoostModelPerSegment(), [LagTransform(in_column="target", lags=[2, 3])]),
        (LinearPerSegmentModel(), [LagTransform(in_column="target", lags=[2, 3])]),
        (LinearMultiSegmentModel(), [LagTransform(in_column="target", lags=[2, 3])]),
        (ElasticPerSegmentModel(), [LagTransform(in_column="target", lags=[2, 3])]),
        (ElasticMultiSegmentModel(), [LagTransform(in_column="target", lags=[2, 3])]),
    ],
)
def test_forecast_in_sample_full_failed(model, transforms, example_tsds):
    _test_forecast_in_sample_full(example_tsds, model, transforms)


@pytest.mark.parametrize(
    "model, transforms",
    [
        (BATSModel(use_trend=True), []),
        (TBATSModel(use_trend=True), []),
        (
            DeepARModel(max_epochs=1, learning_rate=[0.01]),
            [
                PytorchForecastingTransform(
                    max_encoder_length=1,
                    max_prediction_length=1,
                    time_varying_known_reals=["time_idx"],
                    time_varying_unknown_reals=["target"],
                    target_normalizer=GroupNormalizer(groups=["segment"]),
                )
            ],
        ),
        (
            TFTModel(max_epochs=1, learning_rate=[0.01]),
            [
                PytorchForecastingTransform(
                    max_encoder_length=21,
                    min_encoder_length=21,
                    max_prediction_length=5,
                    time_varying_known_reals=["time_idx"],
                    time_varying_unknown_reals=["target"],
                    static_categoricals=["segment"],
                    target_normalizer=None,
                )
            ],
        ),
    ],
)
def test_forecast_in_sample_full_not_implemented(model, transforms, example_tsds):
    with pytest.raises(NotImplementedError, match="It is not possible to make in-sample predictions"):
        _test_forecast_in_sample_full(example_tsds, model, transforms)


@pytest.mark.parametrize(
    "model, transforms",
    [
        (CatBoostModelPerSegment(), [LagTransform(in_column="target", lags=[2, 3])]),
        (CatBoostModelMultiSegment(), [LagTransform(in_column="target", lags=[2, 3])]),
        (LinearPerSegmentModel(), [LagTransform(in_column="target", lags=[2, 3])]),
        (LinearMultiSegmentModel(), [LagTransform(in_column="target", lags=[2, 3])]),
        (ElasticPerSegmentModel(), [LagTransform(in_column="target", lags=[2, 3])]),
        (ElasticMultiSegmentModel(), [LagTransform(in_column="target", lags=[2, 3])]),
        (ProphetModel(), []),
        (SARIMAXModel(), []),
        (AutoARIMAModel(), []),
        (HoltModel(), []),
        (HoltWintersModel(), []),
        (SimpleExpSmoothingModel(), []),
        (MovingAverageModel(window=3), []),
        (NaiveModel(lag=3), []),
        (SeasonalMovingAverageModel(), []),
    ],
)
def test_forecast_in_sample_suffix(model, transforms, example_tsds):
    _test_forecast_in_sample_suffix(example_tsds, model, transforms)


@pytest.mark.parametrize(
    "model, transforms",
    [
        (BATSModel(use_trend=True), []),
        (TBATSModel(use_trend=True), []),
        (
            DeepARModel(max_epochs=1, learning_rate=[0.01]),
            [
                PytorchForecastingTransform(
                    max_encoder_length=1,
                    max_prediction_length=1,
                    time_varying_known_reals=["time_idx"],
                    time_varying_unknown_reals=["target"],
                    target_normalizer=GroupNormalizer(groups=["segment"]),
                )
            ],
        ),
        (
            TFTModel(max_epochs=1, learning_rate=[0.01]),
            [
                PytorchForecastingTransform(
                    max_encoder_length=21,
                    min_encoder_length=21,
                    max_prediction_length=5,
                    time_varying_known_reals=["time_idx"],
                    time_varying_unknown_reals=["target"],
                    static_categoricals=["segment"],
                    target_normalizer=None,
                )
            ],
        ),
    ],
)
def test_forecast_in_sample_suffix_not_implemented(model, transforms, example_tsds):
    with pytest.raises(NotImplementedError, match="It is not possible to make in-sample predictions"):
        _test_forecast_in_sample_suffix(example_tsds, model, transforms)


@pytest.mark.parametrize(
    "model, transforms",
    [
        (CatBoostModelPerSegment(), [LagTransform(in_column="target", lags=[5, 6])]),
        (CatBoostModelMultiSegment(), [LagTransform(in_column="target", lags=[5, 6])]),
        (LinearPerSegmentModel(), [LagTransform(in_column="target", lags=[5, 6])]),
        (LinearMultiSegmentModel(), [LagTransform(in_column="target", lags=[5, 6])]),
        (ElasticPerSegmentModel(), [LagTransform(in_column="target", lags=[5, 6])]),
        (ElasticMultiSegmentModel(), [LagTransform(in_column="target", lags=[5, 6])]),
        (AutoARIMAModel(), []),
        (ProphetModel(), []),
        (SARIMAXModel(), []),
        (HoltModel(), []),
        (HoltWintersModel(), []),
        (SimpleExpSmoothingModel(), []),
        (MovingAverageModel(window=3), []),
        (SeasonalMovingAverageModel(), []),
        (NaiveModel(lag=3), []),
        (BATSModel(use_trend=True), []),
        (TBATSModel(use_trend=True), []),
        (
            DeepARModel(max_epochs=5, learning_rate=[0.01]),
            [
                PytorchForecastingTransform(
                    max_encoder_length=5,
                    max_prediction_length=5,
                    time_varying_known_reals=["time_idx"],
                    time_varying_unknown_reals=["target"],
                    target_normalizer=GroupNormalizer(groups=["segment"]),
                )
            ],
        ),
        (
            TFTModel(max_epochs=1, learning_rate=[0.01]),
            [
                PytorchForecastingTransform(
                    max_encoder_length=21,
                    min_encoder_length=21,
                    max_prediction_length=5,
                    time_varying_known_reals=["time_idx"],
                    time_varying_unknown_reals=["target"],
                    static_categoricals=["segment"],
                    target_normalizer=None,
                )
            ],
        ),
    ],
)
def test_forecast_out_sample_prefix(model, transforms, example_tsds):
    _test_forecast_out_sample_prefix(example_tsds, model, transforms)


@pytest.mark.parametrize(
    "model, transforms",
    [
        (CatBoostModelPerSegment(), [LagTransform(in_column="target", lags=[5, 6])]),
        (CatBoostModelMultiSegment(), [LagTransform(in_column="target", lags=[5, 6])]),
        (LinearPerSegmentModel(), [LagTransform(in_column="target", lags=[5, 6])]),
        (LinearMultiSegmentModel(), [LagTransform(in_column="target", lags=[5, 6])]),
        (ElasticPerSegmentModel(), [LagTransform(in_column="target", lags=[5, 6])]),
        (ElasticMultiSegmentModel(), [LagTransform(in_column="target", lags=[5, 6])]),
        (ProphetModel(), []),
        (SARIMAXModel(), []),
        (HoltModel(), []),
        (HoltWintersModel(), []),
        (SimpleExpSmoothingModel(), []),
        (BATSModel(use_trend=True), []),
        (TBATSModel(use_trend=True), []),
    ],
)
def test_forecast_out_sample_suffix(model, transforms, example_tsds):
    _test_forecast_out_sample_suffix(example_tsds, model, transforms)


@pytest.mark.parametrize(
    "model, transforms",
    [
        (
            TFTModel(max_epochs=1, learning_rate=[0.01]),
            [
                PytorchForecastingTransform(
                    max_encoder_length=21,
                    min_encoder_length=21,
                    max_prediction_length=5,
                    time_varying_known_reals=["time_idx"],
                    time_varying_unknown_reals=["target"],
                    static_categoricals=["segment"],
                    target_normalizer=None,
                )
            ],
        ),
        (
            DeepARModel(max_epochs=5, learning_rate=[0.01]),
            [
                PytorchForecastingTransform(
                    max_encoder_length=5,
                    max_prediction_length=5,
                    time_varying_known_reals=["time_idx"],
                    time_varying_unknown_reals=["target"],
                    target_normalizer=GroupNormalizer(groups=["segment"]),
                )
            ],
        ),
    ],
)
def test_forecast_out_sample_suffix_not_implemented(model, transforms, example_tsds):
    with pytest.raises(NotImplementedError, match="You can only forecast from the next point after the last one"):
        _test_forecast_out_sample_suffix(example_tsds, model, transforms)


@pytest.mark.xfail(strict=True)
@pytest.mark.parametrize(
    "model, transforms",
    [
        (AutoARIMAModel(), []),
        (MovingAverageModel(window=3), []),
        (SeasonalMovingAverageModel(), []),
        (NaiveModel(lag=3), []),
    ],
)
def test_forecast_out_sample_suffix_failed(model, transforms, example_tsds):
    _test_forecast_out_sample_suffix(example_tsds, model, transforms)


@pytest.mark.parametrize(
    "model, transforms",
    [
        (CatBoostModelPerSegment(), [LagTransform(in_column="target", lags=[5, 6])]),
        (CatBoostModelMultiSegment(), [LagTransform(in_column="target", lags=[5, 6])]),
        (LinearPerSegmentModel(), [LagTransform(in_column="target", lags=[5, 6])]),
        (LinearMultiSegmentModel(), [LagTransform(in_column="target", lags=[5, 6])]),
        (ElasticPerSegmentModel(), [LagTransform(in_column="target", lags=[5, 6])]),
        (ElasticMultiSegmentModel(), [LagTransform(in_column="target", lags=[5, 6])]),
        (ProphetModel(), []),
        (HoltModel(), []),
        (HoltWintersModel(), []),
        (SimpleExpSmoothingModel(), []),
    ],
)
def test_forecast_mixed_in_out_sample(model, transforms, example_tsds):
    _test_forecast_mixed_in_out_sample(example_tsds, model, transforms)


@pytest.mark.xfail(strict=True)
@pytest.mark.parametrize(
    "model, transforms",
    [
        (SARIMAXModel(), []),
        (AutoARIMAModel(), []),
    ],
)
def test_forecast_mixed_in_out_sample_failed(model, transforms, example_tsds):
    _test_forecast_mixed_in_out_sample(example_tsds, model, transforms)


@pytest.mark.parametrize(
    "model, transforms",
    [
        (BATSModel(use_trend=True), []),
        (TBATSModel(use_trend=True), []),
        (
            DeepARModel(max_epochs=5, learning_rate=[0.01]),
            [
                PytorchForecastingTransform(
                    max_encoder_length=5,
                    max_prediction_length=5,
                    time_varying_known_reals=["time_idx"],
                    time_varying_unknown_reals=["target"],
                    target_normalizer=GroupNormalizer(groups=["segment"]),
                )
            ],
        ),
        (
            TFTModel(max_epochs=1, learning_rate=[0.01]),
            [
                PytorchForecastingTransform(
                    max_encoder_length=21,
                    min_encoder_length=21,
                    max_prediction_length=5,
                    time_varying_known_reals=["time_idx"],
                    time_varying_unknown_reals=["target"],
                    static_categoricals=["segment"],
                    target_normalizer=None,
                )
            ],
        ),
    ],
)
def test_forecast_mixed_in_out_sample_not_implemented(model, transforms, example_tsds):
    with pytest.raises(NotImplementedError, match="It is not possible to make in-sample predictions"):
        _test_forecast_mixed_in_out_sample(example_tsds, model, transforms)
