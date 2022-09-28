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
from etna.models.nn import RNNModel
from etna.models.nn import TFTModel
from etna.transforms import LagTransform
from etna.transforms import PytorchForecastingTransform
from tests.test_models.test_inference.common import _test_prediction_in_sample_full
from tests.test_models.test_inference.common import _test_prediction_in_sample_suffix
from tests.test_models.test_inference.common import make_prediction
from tests.test_models.test_inference.common import to_be_fixed


def make_predict(model, ts, prediction_size) -> TSDataset:
    return make_prediction(model=model, ts=ts, prediction_size=prediction_size, method_name="predict")


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
    def test_predict_in_sample_full(self, model, transforms, example_tsds):
        _test_prediction_in_sample_full(example_tsds, model, transforms, method_name="predict")

    @pytest.mark.parametrize(
        "model, transforms",
        [
            (LinearPerSegmentModel(), [LagTransform(in_column="target", lags=[2, 3])]),
            (LinearMultiSegmentModel(), [LagTransform(in_column="target", lags=[2, 3])]),
            (ElasticPerSegmentModel(), [LagTransform(in_column="target", lags=[2, 3])]),
            (ElasticMultiSegmentModel(), [LagTransform(in_column="target", lags=[2, 3])]),
        ],
    )
    def test_predict_in_sample_full_failed_not_enough_context(self, model, transforms, example_tsds):
        with pytest.raises(ValueError, match="Input contains NaN, infinity or a value too large"):
            _test_prediction_in_sample_full(example_tsds, model, transforms, method_name="predict")

    @pytest.mark.parametrize(
        "model, transforms",
        [
            (MovingAverageModel(window=3), []),
            (NaiveModel(lag=3), []),
            (SeasonalMovingAverageModel(), []),
            (DeadlineMovingAverageModel(window=1), []),
        ],
    )
    def test_predict_in_sample_full_failed_not_enough_context(self, model, transforms, example_tsds):
        with pytest.raises(ValueError, match="Given context isn't big enough"):
            _test_prediction_in_sample_full(example_tsds, model, transforms, method_name="predict")

    @to_be_fixed(raises=NotImplementedError, match="Method predict isn't currently implemented")
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
            (RNNModel(input_size=1, encoder_length=7, decoder_length=7, trainer_params=dict(max_epochs=1)), []),
        ],
    )
    def test_predict_in_sample_full_failed_not_implemented_predict(self, model, transforms, example_tsds):
        _test_prediction_in_sample_full(example_tsds, model, transforms, method_name="predict")

    @to_be_fixed(raises=NotImplementedError, match="It is not possible to make in-sample predictions")
    @pytest.mark.parametrize(
        "model, transforms",
        [],
    )
    def test_predict_in_sample_full_failed_not_implemented_in_sample(self, model, transforms, example_tsds):
        _test_prediction_in_sample_full(example_tsds, model, transforms, method_name="predict")


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
            (MovingAverageModel(window=3), []),
            (NaiveModel(lag=3), []),
            (SeasonalMovingAverageModel(), []),
            (DeadlineMovingAverageModel(window=1), []),
        ],
    )
    def test_predict_in_sample_suffix(self, model, transforms, example_tsds):
        _test_prediction_in_sample_suffix(example_tsds, model, transforms, method_name="predict", num_skip_points=50)

    @to_be_fixed(raises=NotImplementedError, match="Method predict isn't currently implemented")
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
            (RNNModel(input_size=1, encoder_length=7, decoder_length=7, trainer_params=dict(max_epochs=1)), []),
        ],
    )
    def test_predict_in_sample_full_failed_not_implemented_predict(self, model, transforms, example_tsds):
        _test_prediction_in_sample_suffix(example_tsds, model, transforms, method_name="predict", num_skip_points=50)

    @to_be_fixed(raises=NotImplementedError, match="It is not possible to make in-sample predictions")
    @pytest.mark.parametrize(
        "model, transforms",
        [],
    )
    def test_predict_in_sample_suffix_failed_not_implemented_in_sample(self, model, transforms, example_tsds):
        _test_prediction_in_sample_suffix(example_tsds, model, transforms, method_name="predict", num_skip_points=50)


class TestPredictOutSample:
    """Test predict on future dataset.

    Expected that target values are filled after prediction.
    """

    @staticmethod
    def _test_predict_out_sample(ts, model, transforms, prediction_size=5):
        train_ts, future_ts = ts.train_test_split(test_size=prediction_size)
        forecast_ts = TSDataset(df=ts.df, freq=ts.freq)

        # fitting
        train_ts.fit_transform(transforms)
        model.fit(train_ts)

        # forecasting
        forecast_ts.transform(train_ts.transforms)
        to_remain = model.context_size + prediction_size
        forecast_ts.df = forecast_ts.df.iloc[-to_remain:]
        forecast_ts = make_predict(model=model, ts=forecast_ts, prediction_size=prediction_size)

        # checking
        forecast_df = forecast_ts.to_pandas(flatten=True)
        assert not np.any(forecast_df["target"].isna())

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
    def test_predict_out_sample(self, model, transforms, example_tsds):
        self._test_predict_out_sample(example_tsds, model, transforms)

    @to_be_fixed(raises=NotImplementedError, match="Method predict isn't currently implemented")
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
            (RNNModel(input_size=1, encoder_length=7, decoder_length=7, trainer_params=dict(max_epochs=1)), []),
        ],
    )
    def test_predict_out_sample_failed_not_implemented_predict(self, model, transforms, example_tsds):
        self._test_predict_out_sample(example_tsds, model, transforms)


class TestPredictMixedInOutSample:
    """Test predict on mixture of in-sample and out-sample.

    Expected that predictions on in-sample and out-sample separately match predictions on full mixed dataset.
    """

    @staticmethod
    def _test_predict_mixed_in_out_sample(ts, model, transforms, num_skip_points=50, future_prediction_size=5):
        train_ts, future_ts = ts.train_test_split(test_size=future_prediction_size)
        train_df = train_ts.to_pandas()
        future_df = future_ts.to_pandas()
        train_ts.fit_transform(transforms)
        model.fit(train_ts)

        # predicting mixed in-sample and out-sample
        df_full = pd.concat((train_df, future_df))
        forecast_full_ts = TSDataset(df=df_full, freq=ts.freq)
        forecast_full_ts.transform(train_ts.transforms)
        forecast_full_ts.df = forecast_full_ts.df.iloc[(num_skip_points - model.context_size) :]
        full_prediction_size = len(forecast_full_ts.index) - model.context_size
        forecast_full_ts = make_predict(model=model, ts=forecast_full_ts, prediction_size=full_prediction_size)

        # predicting only in sample
        forecast_in_sample_ts = TSDataset(train_df, freq=ts.freq)
        forecast_in_sample_ts.transform(train_ts.transforms)
        to_skip = num_skip_points - model.context_size
        forecast_in_sample_ts.df = forecast_in_sample_ts.df.iloc[to_skip:]
        in_sample_prediction_size = len(forecast_in_sample_ts.index) - model.context_size
        forecast_in_sample_ts = make_predict(
            model=model, ts=forecast_in_sample_ts, prediction_size=in_sample_prediction_size
        )

        # predicting only out sample
        forecast_out_sample_ts = TSDataset(df=df_full, freq=ts.freq)
        forecast_out_sample_ts.transform(train_ts.transforms)
        to_remain = model.context_size + future_prediction_size
        forecast_out_sample_ts.df = forecast_out_sample_ts.df.iloc[-to_remain:]
        forecast_out_sample_ts = make_predict(
            model=model, ts=forecast_out_sample_ts, prediction_size=future_prediction_size
        )

        # checking
        forecast_full_df = forecast_full_ts.to_pandas()
        forecast_in_sample_df = forecast_in_sample_ts.to_pandas()
        forecast_out_sample_df = forecast_out_sample_ts.to_pandas()
        assert_frame_equal(forecast_in_sample_df, forecast_full_df.iloc[:-future_prediction_size])
        assert_frame_equal(forecast_out_sample_df, forecast_full_df.iloc[-future_prediction_size:])

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
    def test_predict_mixed_in_out_sample(self, model, transforms, example_tsds):
        self._test_predict_mixed_in_out_sample(example_tsds, model, transforms)

    @to_be_fixed(raises=NotImplementedError, match="Method predict isn't currently implemented")
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
            (RNNModel(input_size=1, encoder_length=7, decoder_length=7, trainer_params=dict(max_epochs=1)), []),
        ],
    )
    def test_predict_mixed_in_out_sample_failed_not_implemented_predict(self, model, transforms, example_tsds):
        self._test_predict_mixed_in_out_sample(example_tsds, model, transforms)
