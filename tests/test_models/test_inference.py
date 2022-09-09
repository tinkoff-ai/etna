from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from pandas.util.testing import assert_frame_equal
from pytorch_forecasting.data import GroupNormalizer
from typing_extensions import get_args

from etna.datasets import TSDataset
from etna.models import AutoARIMAModel
from etna.models import BATSModel
from etna.models import CatBoostModelMultiSegment
from etna.models import CatBoostModelPerSegment
from etna.models import ContextRequiredModelType
from etna.models import DeadlineMovingAverageModel
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


def _test_forecast_in_sample_full_no_target(ts, model, transforms):
    df = ts.to_pandas()

    # fitting
    ts.fit_transform(transforms)
    model.fit(ts)

    # forecasting
    forecast_ts = TSDataset(df, freq="D")
    forecast_ts.transform(ts.transforms)
    forecast_ts.df.loc[:, pd.IndexSlice[:, "target"]] = np.NaN

    if isinstance(model, get_args(ContextRequiredModelType)):
        prediction_size = len(forecast_ts.index)
        model.forecast(forecast_ts, prediction_size=prediction_size)
    else:
        model.forecast(forecast_ts)

    # checking
    forecast_df = forecast_ts.to_pandas(flatten=True)
    assert not np.any(forecast_df["target"].isna())


def _test_prediction_in_sample_full(ts, model, transforms, method_name):
    df = ts.to_pandas()
    method = getattr(model, method_name)

    # fitting
    ts.fit_transform(transforms)
    model.fit(ts)

    # forecasting
    forecast_ts = TSDataset(df, freq="D")
    forecast_ts.transform(ts.transforms)

    if isinstance(model, get_args(ContextRequiredModelType)):
        prediction_size = len(forecast_ts.index)
        method(forecast_ts, prediction_size=prediction_size)
    else:
        method(forecast_ts)

    # checking
    forecast_df = forecast_ts.to_pandas(flatten=True)
    assert not np.any(forecast_df["target"].isna())


def _test_forecast_in_sample_suffix_no_target(ts, model, transforms, num_skip_points):
    df = ts.to_pandas()

    # fitting
    ts.fit_transform(transforms)
    model.fit(ts)

    # forecasting
    forecast_ts = TSDataset(df, freq="D")
    forecast_ts.transform(ts.transforms)

    if isinstance(model, get_args(ContextRequiredModelType)):
        prediction_size = len(forecast_ts.index) - num_skip_points
        forecast_ts.df.loc[forecast_ts.index[num_skip_points] :, pd.IndexSlice[:, "target"]] = np.NaN
        model.forecast(forecast_ts, prediction_size=prediction_size)
    else:
        forecast_ts.df = forecast_ts.df.iloc[num_skip_points:]
        forecast_ts.df.loc[:, pd.IndexSlice[:, "target"]] = np.NaN
        model.forecast(forecast_ts)

    # checking
    forecast_df = forecast_ts.to_pandas(flatten=True)
    assert not np.any(forecast_df["target"].isna())


def _test_prediction_in_sample_suffix(ts, model, transforms, method_name, num_skip_points):
    df = ts.to_pandas()
    method = getattr(model, method_name)

    # fitting
    ts.fit_transform(transforms)
    model.fit(ts)

    # forecasting
    forecast_ts = TSDataset(df, freq="D")
    forecast_ts.transform(ts.transforms)

    if isinstance(model, get_args(ContextRequiredModelType)):
        prediction_size = len(forecast_ts.index) - num_skip_points
        method(forecast_ts, prediction_size=prediction_size)
    else:
        forecast_ts.df = forecast_ts.df.iloc[num_skip_points:]
        method(forecast_ts)

    # checking
    forecast_df = forecast_ts.to_pandas(flatten=True)
    assert not np.any(forecast_df["target"].isna())


def _test_forecast_out_sample_prefix(ts, model, transforms):
    full_prediction_size = 5
    prefix_prediction_size = 3
    prediction_size_diff = full_prediction_size - prefix_prediction_size

    # fitting
    ts.fit_transform(transforms)
    model.fit(ts)

    # forecasting full
    import torch  # TODO: remove after fix at issue-802

    torch.manual_seed(11)

    if isinstance(model, get_args(ContextRequiredModelType)):
        forecast_full_ts = ts.make_future(future_steps=full_prediction_size, tail_steps=model.context_size)
        model.forecast(forecast_full_ts, prediction_size=full_prediction_size)
    else:
        forecast_full_ts = ts.make_future(future_steps=full_prediction_size)
        model.forecast(forecast_full_ts)

    # forecasting only prefix
    torch.manual_seed(11)  # TODO: remove after fix at issue-802

    if isinstance(model, get_args(ContextRequiredModelType)):
        forecast_prefix_ts = ts.make_future(future_steps=full_prediction_size, tail_steps=model.context_size)
        forecast_prefix_ts.df = forecast_prefix_ts.df.iloc[:-prediction_size_diff]
        model.forecast(forecast_prefix_ts, prediction_size=prefix_prediction_size)
    else:
        forecast_prefix_ts = ts.make_future(future_steps=full_prediction_size)
        forecast_prefix_ts.df = forecast_prefix_ts.df.iloc[:-prediction_size_diff]
        model.forecast(forecast_prefix_ts)

    # checking
    forecast_full_df = forecast_full_ts.to_pandas()
    forecast_prefix_df = forecast_prefix_ts.to_pandas()
    assert_frame_equal(forecast_prefix_df, forecast_full_df.iloc[:prefix_prediction_size])


def _test_forecast_out_sample_suffix(ts, model, transforms):
    full_prediction_size = 5
    suffix_prediction_size = 3
    prediction_size_diff = full_prediction_size - suffix_prediction_size

    # fitting
    ts.fit_transform(transforms)
    model.fit(ts)

    # forecasting full
    if isinstance(model, get_args(ContextRequiredModelType)):
        forecast_full_ts = ts.make_future(future_steps=full_prediction_size, tail_steps=model.context_size)
        model.forecast(forecast_full_ts, prediction_size=full_prediction_size)
    else:
        forecast_full_ts = ts.make_future(future_steps=full_prediction_size)
        model.forecast(forecast_full_ts)

    # forecasting only suffix
    if isinstance(model, get_args(ContextRequiredModelType)):
        forecast_gap_ts = ts.make_future(future_steps=full_prediction_size, tail_steps=model.context_size)

        # firstly we should forecast prefix to use it as a context
        forecast_prefix_ts = deepcopy(forecast_gap_ts)
        forecast_prefix_ts.df = forecast_prefix_ts.df.iloc[:-suffix_prediction_size]
        model.forecast(forecast_prefix_ts, prediction_size=prediction_size_diff)
        forecast_gap_ts.df = forecast_gap_ts.df.combine_first(forecast_prefix_ts.df)

        # forecast suffix with known context for it
        model.forecast(forecast_gap_ts, prediction_size=suffix_prediction_size)
    else:
        forecast_gap_ts = ts.make_future(future_steps=full_prediction_size)
        forecast_gap_ts.df = forecast_gap_ts.df.iloc[prediction_size_diff:]
        model.forecast(forecast_gap_ts)

    # checking
    forecast_full_df = forecast_full_ts.to_pandas()
    forecast_gap_df = forecast_gap_ts.to_pandas()
    assert_frame_equal(forecast_gap_df, forecast_full_df.iloc[prediction_size_diff:])


def _test_predict_out_sample(ts, model, transforms):
    prediction_size = 5

    # fitting
    ts.fit_transform(transforms)
    model.fit(ts)

    # forecasting
    if isinstance(model, get_args(ContextRequiredModelType)):
        forecast_ts = ts.make_future(future_steps=prediction_size, tail_steps=model.context_size)
        model.predict(forecast_ts, prediction_size=prediction_size)
    else:
        forecast_ts = ts.make_future(future_steps=prediction_size)
        model.predict(forecast_ts)

    # checking
    forecast_df = forecast_ts.to_pandas(flatten=True)
    assert not np.any(forecast_df["target"].isna())


def _test_forecast_mixed_in_out_sample(ts, model, transforms, num_skip_points):
    prediction_size = 5

    # fitting
    df = ts.to_pandas()
    ts.fit_transform(transforms)
    model.fit(ts)

    # forecasting mixed in-sample and out-sample
    future_ts = ts.make_future(future_steps=prediction_size)
    future_df = future_ts.to_pandas().loc[:, pd.IndexSlice[:, "target"]]
    df_full = pd.concat((df, future_df))
    forecast_full_ts = TSDataset(df=df_full, freq=ts.freq)
    forecast_full_ts.transform(ts.transforms)
    if isinstance(model, get_args(ContextRequiredModelType)):
        to_skip = num_skip_points - model.context_size
        forecast_full_ts.df = forecast_full_ts.df.iloc[to_skip:]
        prediction_size = len(forecast_full_ts.index) - model.context_size
        model.forecast(forecast_full_ts, prediction_size=prediction_size)
    else:
        forecast_full_ts.df = forecast_full_ts.df.iloc[num_skip_points:]
        model.forecast(forecast_full_ts)

    # checking
    forecast_full_df = forecast_full_ts.to_pandas(flatten=True)
    assert not np.any(forecast_full_df["target"].isna())


def _test_predict_mixed_in_out_sample(ts, model, transforms, num_skip_points):
    prediction_size = 5

    train_ts, future_ts = ts.train_test_split(test_size=prediction_size)
    train_df = train_ts.to_pandas()
    future_df = future_ts.to_pandas()
    train_ts.fit_transform(transforms)
    model.fit(train_ts)

    # predicting mixed in-sample and out-sample
    df_full = pd.concat((train_df, future_df))
    forecast_full_ts = TSDataset(df=df_full, freq=ts.freq)
    forecast_full_ts.transform(train_ts.transforms)
    if isinstance(model, get_args(ContextRequiredModelType)):
        to_skip = num_skip_points - model.context_size
        forecast_full_ts.df = forecast_full_ts.df.iloc[to_skip:]
        prediction_size = len(forecast_full_ts.index) - model.context_size
        model.predict(forecast_full_ts, prediction_size=prediction_size)
    else:
        forecast_full_ts.df = forecast_full_ts.df.iloc[num_skip_points:]
        model.predict(forecast_full_ts)

    # predicting only in sample
    forecast_in_sample_ts = TSDataset(train_df, freq=ts.freq)
    forecast_in_sample_ts.transform(train_ts.transforms)
    if isinstance(model, get_args(ContextRequiredModelType)):
        to_skip = num_skip_points - model.context_size
        forecast_in_sample_ts.df = forecast_in_sample_ts.df.iloc[to_skip:]
        prediction_size = len(forecast_in_sample_ts.index) - model.context_size
        model.predict(forecast_in_sample_ts, prediction_size=prediction_size)
    else:
        forecast_in_sample_ts.df = forecast_in_sample_ts.df.iloc[num_skip_points:]
        model.predict(forecast_in_sample_ts)

    # predicting only out sample
    forecast_out_sample_ts = TSDataset(df=df_full, freq=ts.freq)
    forecast_out_sample_ts.transform(train_ts.transforms)
    if isinstance(model, get_args(ContextRequiredModelType)):
        to_remain = model.context_size + prediction_size
        forecast_out_sample_ts.df = forecast_out_sample_ts.df.iloc[-to_remain:]
        model.predict(forecast_out_sample_ts, prediction_size=prediction_size)
    else:
        forecast_out_sample_ts.df = forecast_out_sample_ts.df.iloc[-prediction_size:]
        model.predict(forecast_out_sample_ts)

    # checking
    forecast_full_df = forecast_full_ts.to_pandas()
    forecast_in_sample_df = forecast_in_sample_ts.to_pandas()
    forecast_out_sample_df = forecast_out_sample_ts.to_pandas()
    assert_frame_equal(forecast_in_sample_df, forecast_full_df.iloc[:-prediction_size])
    assert_frame_equal(forecast_out_sample_df, forecast_full_df.iloc[-prediction_size:])


class TestForecastInSampleFullNoTarget:
    """Test forecast on full train dataset with filling target with NaNs.

    Expected that NaNs are filled after prediction.
    """

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
        ],
    )
    def test_forecast_in_sample_full_no_target(self, model, transforms, example_tsds):
        _test_forecast_in_sample_full_no_target(example_tsds, model, transforms)

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
    def test_forecast_in_sample_full_no_target_failed(self, model, transforms, example_tsds):
        _test_forecast_in_sample_full_no_target(example_tsds, model, transforms)

    @pytest.mark.parametrize(
        "model, transforms",
        [
            (MovingAverageModel(window=3), []),
            (NaiveModel(lag=3), []),
            (SeasonalMovingAverageModel(), []),
            (DeadlineMovingAverageModel(window=1), []),
        ],
    )
    def test_forecast_in_sample_full_no_target_failed_not_enough_context(self, model, transforms, example_tsds):
        with pytest.raises(ValueError, match="Given context isn't big enough"):
            _test_forecast_in_sample_full_no_target(example_tsds, model, transforms)

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
    def test_forecast_in_sample_full_no_target_not_implemented_in_sample(self, model, transforms, example_tsds):
        with pytest.raises(NotImplementedError, match="It is not possible to make in-sample predictions"):
            _test_forecast_in_sample_full_no_target(example_tsds, model, transforms)


class TestForecastInSampleFull:
    """Test forecast on full train dataset.

    Expected that target values are filled after prediction.
    """

    @pytest.mark.parametrize(
        "model, transforms",
        [
            (CatBoostModelMultiSegment(), [LagTransform(in_column="target", lags=[2, 3])]),
            (CatBoostModelPerSegment(), [LagTransform(in_column="target", lags=[2, 3])]),
            (ProphetModel(), []),
            (SARIMAXModel(), []),
            (AutoARIMAModel(), []),
            (HoltModel(), []),
            (HoltWintersModel(), []),
            (SimpleExpSmoothingModel(), []),
        ],
    )
    def test_forecast_in_sample_full(self, model, transforms, example_tsds):
        _test_prediction_in_sample_full(example_tsds, model, transforms, method_name="forecast")

    @pytest.mark.xfail(strict=True)
    @pytest.mark.parametrize(
        "model, transforms",
        [
            (LinearPerSegmentModel(), [LagTransform(in_column="target", lags=[2, 3])]),
            (LinearMultiSegmentModel(), [LagTransform(in_column="target", lags=[2, 3])]),
            (ElasticPerSegmentModel(), [LagTransform(in_column="target", lags=[2, 3])]),
            (ElasticMultiSegmentModel(), [LagTransform(in_column="target", lags=[2, 3])]),
        ],
    )
    def test_forecast_in_sample_full_failed(self, model, transforms, example_tsds):
        _test_prediction_in_sample_full(example_tsds, model, transforms, method_name="forecast")

    @pytest.mark.parametrize(
        "model, transforms",
        [
            (MovingAverageModel(window=3), []),
            (NaiveModel(lag=3), []),
            (SeasonalMovingAverageModel(), []),
            (DeadlineMovingAverageModel(window=1), []),
        ],
    )
    def test_forecast_in_sample_full_failed_not_enough_context(self, model, transforms, example_tsds):
        with pytest.raises(ValueError, match="Given context isn't big enough"):
            _test_prediction_in_sample_full(example_tsds, model, transforms, method_name="forecast")

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
    def test_forecast_in_sample_full_not_implemented(self, model, transforms, example_tsds):
        with pytest.raises(NotImplementedError, match="It is not possible to make in-sample predictions"):
            _test_prediction_in_sample_full(example_tsds, model, transforms, method_name="forecast")


class TestPredictInSampleFull:
    """Test predict on full train dataset.

    Expected that target values are filled after prediction.
    """

    @pytest.mark.parametrize(
        "model, transforms",
        [
            (CatBoostModelMultiSegment(), [LagTransform(in_column="target", lags=[2, 3])]),
            (CatBoostModelPerSegment(), [LagTransform(in_column="target", lags=[2, 3])]),
            (ProphetModel(), []),
            (SARIMAXModel(), []),
            (AutoARIMAModel(), []),
            (HoltModel(), []),
            (HoltWintersModel(), []),
            (SimpleExpSmoothingModel(), []),
        ],
    )
    def test_forecast_in_sample_full(self, model, transforms, example_tsds):
        _test_prediction_in_sample_full(example_tsds, model, transforms, method_name="predict")

    @pytest.mark.xfail(strict=True)
    @pytest.mark.parametrize(
        "model, transforms",
        [
            (LinearPerSegmentModel(), [LagTransform(in_column="target", lags=[2, 3])]),
            (LinearMultiSegmentModel(), [LagTransform(in_column="target", lags=[2, 3])]),
            (ElasticPerSegmentModel(), [LagTransform(in_column="target", lags=[2, 3])]),
            (ElasticMultiSegmentModel(), [LagTransform(in_column="target", lags=[2, 3])]),
        ],
    )
    def test_forecast_in_sample_full_failed(self, model, transforms, example_tsds):
        _test_prediction_in_sample_full(example_tsds, model, transforms, method_name="predict")

    @pytest.mark.parametrize(
        "model, transforms",
        [],
    )
    def test_forecast_in_sample_full_failed_not_enough_context(self, model, transforms, example_tsds):
        with pytest.raises(ValueError, match="Given context isn't big enough"):
            _test_prediction_in_sample_full(example_tsds, model, transforms, method_name="predict")

    @pytest.mark.parametrize(
        "model, transforms",
        [
            (MovingAverageModel(window=3), []),
            (NaiveModel(lag=3), []),
            (SeasonalMovingAverageModel(), []),
            (DeadlineMovingAverageModel(window=1), []),
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
    def test_forecast_in_sample_full_failed_not_implemented_predict(self, model, transforms, example_tsds):
        with pytest.raises(NotImplementedError, match="Method predict isn't currently implemented"):
            _test_prediction_in_sample_full(example_tsds, model, transforms, method_name="predict")

    @pytest.mark.parametrize(
        "model, transforms",
        [],
    )
    def test_forecast_in_sample_full_failed_not_implemented_in_sample(self, model, transforms, example_tsds):
        with pytest.raises(NotImplementedError, match="It is not possible to make in-sample predictions"):
            _test_prediction_in_sample_full(example_tsds, model, transforms, method_name="predict")


class TestForecastInSampleSuffixNoTarget:
    """Test forecast on suffix of train dataset with filling target with NaNs.

    Expected that NaNs are filled after prediction.
    """

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
            (DeadlineMovingAverageModel(window=1), []),
        ],
    )
    def test_forecast_in_sample_suffix_no_target(self, model, transforms, example_tsds):
        _test_forecast_in_sample_suffix_no_target(example_tsds, model, transforms, num_skip_points=50)

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
    def test_forecast_in_sample_suffix_no_target_failed_not_implemented_in_sample(
        self, model, transforms, example_tsds
    ):
        with pytest.raises(NotImplementedError, match="It is not possible to make in-sample predictions"):
            _test_forecast_in_sample_suffix_no_target(example_tsds, model, transforms, num_skip_points=50)


class TestForecastInSampleSuffix:
    """Test forecast on suffix of train dataset.

    Expected that target values are filled after prediction.
    """

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
            (DeadlineMovingAverageModel(window=1), []),
        ],
    )
    def test_forecast_in_sample_suffix(self, model, transforms, example_tsds):
        _test_prediction_in_sample_suffix(example_tsds, model, transforms, method_name="forecast", num_skip_points=50)

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
    def test_forecast_in_sample_suffix_failed_not_implemented_in_sample(self, model, transforms, example_tsds):
        with pytest.raises(NotImplementedError, match="It is not possible to make in-sample predictions"):
            _test_prediction_in_sample_suffix(
                example_tsds, model, transforms, method_name="forecast", num_skip_points=50
            )


class TestPredictInSampleSuffix:
    """Test predict on suffix of train dataset.

    Expected that target values are filled after prediction.
    """

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
        ],
    )
    def test_forecast_in_sample_suffix(self, model, transforms, example_tsds):
        _test_prediction_in_sample_suffix(example_tsds, model, transforms, method_name="predict", num_skip_points=50)

    @pytest.mark.parametrize(
        "model, transforms",
        [
            (MovingAverageModel(window=3), []),
            (NaiveModel(lag=3), []),
            (SeasonalMovingAverageModel(), []),
            (DeadlineMovingAverageModel(window=1), []),
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
    def test_forecast_in_sample_full_failed_not_implemented_predict(self, model, transforms, example_tsds):
        with pytest.raises(NotImplementedError, match="Method predict isn't currently implemented"):
            _test_prediction_in_sample_suffix(
                example_tsds, model, transforms, method_name="predict", num_skip_points=50
            )

    @pytest.mark.parametrize(
        "model, transforms",
        [],
    )
    def test_forecast_in_sample_suffix_failed_not_implemented_in_sample(self, model, transforms, example_tsds):
        with pytest.raises(NotImplementedError, match="It is not possible to make in-sample predictions"):
            _test_prediction_in_sample_suffix(
                example_tsds, model, transforms, method_name="predict", num_skip_points=50
            )


class TestForecastOutSamplePrefix:
    """Test forecast on prefix of future dataset.

    Expected that predictions on prefix match prefix of predictions on full future dataset.
    """

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
            (DeadlineMovingAverageModel(window=1), []),
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
    def test_forecast_out_sample_prefix(self, model, transforms, example_tsds):
        _test_forecast_out_sample_prefix(example_tsds, model, transforms)


class TestForecastOutSampleSuffix:
    """Test forecast on suffix of future dataset.

    Expected that predictions on suffix match suffix of predictions on full future dataset.
    """

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
            (BATSModel(use_trend=True), []),
            (TBATSModel(use_trend=True), []),
            (MovingAverageModel(window=3), []),
            (SeasonalMovingAverageModel(), []),
            (NaiveModel(lag=3), []),
            (DeadlineMovingAverageModel(window=1), []),
        ],
    )
    def test_forecast_out_sample_suffix(self, model, transforms, example_tsds):
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
    def test_forecast_out_sample_suffix_failed_not_implemented(self, model, transforms, example_tsds):
        with pytest.raises(NotImplementedError, match="You can only forecast from the next point after the last one"):
            _test_forecast_out_sample_suffix(example_tsds, model, transforms)


class TestPredictOutSample:
    """Test predict on prefix of future dataset.

    Expected that target values are filled after prediction.
    """

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
        ],
    )
    def test_predict_out_sample(self, model, transforms, example_tsds):
        _test_predict_out_sample(example_tsds, model, transforms)

    @pytest.mark.parametrize(
        "model, transforms",
        [
            (MovingAverageModel(window=3), []),
            (SeasonalMovingAverageModel(), []),
            (NaiveModel(lag=3), []),
            (DeadlineMovingAverageModel(window=1), []),
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
    def test_predict_out_sample_failed_not_implemented_predict(self, model, transforms, example_tsds):
        with pytest.raises(NotImplementedError, match="Method predict isn't currently implemented"):
            _test_predict_out_sample(example_tsds, model, transforms)


class TestForecastMixedInOutSample:
    """Test forecast on mixture of in-sample and out-sample.

    Expected that target values are filled after prediction.
    """

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
            (DeadlineMovingAverageModel(window=1), []),
        ],
    )
    def test_forecast_mixed_in_out_sample(self, model, transforms, example_tsds):
        _test_forecast_mixed_in_out_sample(example_tsds, model, transforms, num_skip_points=50)

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
    def test_forecast_mixed_in_out_sample_failed_not_implemented_in_sample(self, model, transforms, example_tsds):
        with pytest.raises(NotImplementedError, match="It is not possible to make in-sample predictions"):
            _test_forecast_mixed_in_out_sample(example_tsds, model, transforms, num_skip_points=50)


class TestPredictMixedInOutSample:
    """Test predict on mixture of in-sample and out-sample.

    Expected that predictions on in-sample and out-sample separately match predictions on full mixed dataset.
    """

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
        ],
    )
    def test_predict_mixed_in_out_sample(self, model, transforms, example_tsds):
        _test_predict_mixed_in_out_sample(example_tsds, model, transforms, num_skip_points=50)

    @pytest.mark.parametrize(
        "model, transforms",
        [
            (MovingAverageModel(window=3), []),
            (SeasonalMovingAverageModel(), []),
            (NaiveModel(lag=3), []),
            (DeadlineMovingAverageModel(window=1), []),
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
    def test_predict_mixed_in_out_sample_failed_not_implemented_predict(self, model, transforms, example_tsds):
        with pytest.raises(NotImplementedError, match="Method predict isn't currently implemented"):
            _test_predict_mixed_in_out_sample(example_tsds, model, transforms, num_skip_points=50)
