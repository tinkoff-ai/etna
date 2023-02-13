from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from pandas.util.testing import assert_frame_equal
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
from etna.models.nn import RNNModel
from etna.models.nn import TFTModel
from etna.models.nn.utils import PytorchForecastingDatasetBuilder
from etna.transforms import LagTransform
from tests.test_models.inference.common import _test_prediction_in_sample_full
from tests.test_models.inference.common import _test_prediction_in_sample_suffix
from tests.test_models.inference.common import make_prediction
from tests.test_models.inference.common import to_be_fixed


def make_forecast(model, ts, prediction_size) -> TSDataset:
    return make_prediction(model=model, ts=ts, prediction_size=prediction_size, method_name="forecast")


class TestForecastInSampleFullNoTarget:
    """Test forecast on full train dataset with filling target with NaNs.

    Expected that NaNs are filled after prediction.
    """

    @staticmethod
    def _test_forecast_in_sample_full_no_target(ts, model, transforms):
        df = ts.to_pandas()

        # fitting
        ts.fit_transform(transforms)
        model.fit(ts)

        # forecasting
        forecast_ts = TSDataset(df, freq="D")
        forecast_ts.transform(ts.transforms)
        forecast_ts.df.loc[:, pd.IndexSlice[:, "target"]] = np.NaN
        prediction_size = len(forecast_ts.index)
        forecast_ts = make_forecast(model=model, ts=forecast_ts, prediction_size=prediction_size)

        # checking
        forecast_df = forecast_ts.to_pandas(flatten=True)
        assert not np.any(forecast_df["target"].isna())

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
    def test_forecast_in_sample_full_no_target(self, model, transforms, example_tsds):
        self._test_forecast_in_sample_full_no_target(example_tsds, model, transforms)

    @to_be_fixed(raises=AssertionError)
    @pytest.mark.parametrize(
        "model, transforms",
        [
            (RNNModel(input_size=1, encoder_length=7, decoder_length=7, trainer_params=dict(max_epochs=1)), []),
        ],
    )
    def test_forecast_in_sample_full_no_target_failed(self, model, transforms, example_tsds):
        self._test_forecast_in_sample_full_no_target(example_tsds, model, transforms)

    @pytest.mark.parametrize(
        "model, transforms",
        [
            (LinearPerSegmentModel(), [LagTransform(in_column="target", lags=[2, 3])]),
            (LinearMultiSegmentModel(), [LagTransform(in_column="target", lags=[2, 3])]),
            (ElasticPerSegmentModel(), [LagTransform(in_column="target", lags=[2, 3])]),
            (ElasticMultiSegmentModel(), [LagTransform(in_column="target", lags=[2, 3])]),
        ],
    )
    def test_forecast_in_sample_full_no_target_failed_nans_lags(self, model, transforms, example_tsds):
        with pytest.raises(ValueError, match="Input contains NaN, infinity or a value too large"):
            self._test_forecast_in_sample_full_no_target(example_tsds, model, transforms)

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
            self._test_forecast_in_sample_full_no_target(example_tsds, model, transforms)

    @to_be_fixed(raises=NotImplementedError, match="It is not possible to make in-sample predictions")
    @pytest.mark.parametrize(
        "model, transforms",
        [
            (BATSModel(use_trend=True), []),
            (TBATSModel(use_trend=True), []),
            (
                DeepARModel(trainer_params=dict(max_epochs=1), lr=0.01, decoder_length=1, encoder_length=1),
                [],
            ),
            (
                TFTModel(
                    max_epochs=1,
                    learning_rate=[0.01],
                    dataset_builder=PytorchForecastingDatasetBuilder(
                        max_encoder_length=21,
                        min_encoder_length=21,
                        max_prediction_length=5,
                        time_varying_known_reals=["time_idx"],
                        time_varying_unknown_reals=["target"],
                        static_categoricals=["segment"],
                        target_normalizer=None,
                    ),
                ),
                [],
            ),
        ],
    )
    def test_forecast_in_sample_full_no_target_failed_not_implemented_in_sample(self, model, transforms, example_tsds):
        self._test_forecast_in_sample_full_no_target(example_tsds, model, transforms)


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
            (RNNModel(input_size=1, encoder_length=7, decoder_length=7, trainer_params=dict(max_epochs=1)), []),
        ],
    )
    def test_forecast_in_sample_full(self, model, transforms, example_tsds):
        _test_prediction_in_sample_full(example_tsds, model, transforms, method_name="forecast")

    @pytest.mark.parametrize(
        "model, transforms",
        [
            (LinearPerSegmentModel(), [LagTransform(in_column="target", lags=[2, 3])]),
            (LinearMultiSegmentModel(), [LagTransform(in_column="target", lags=[2, 3])]),
            (ElasticPerSegmentModel(), [LagTransform(in_column="target", lags=[2, 3])]),
            (ElasticMultiSegmentModel(), [LagTransform(in_column="target", lags=[2, 3])]),
        ],
    )
    def test_forecast_in_sample_full_failed_nans_lags(self, model, transforms, example_tsds):
        with pytest.raises(ValueError, match="Input contains NaN, infinity or a value too large"):
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

    @to_be_fixed(raises=NotImplementedError, match="It is not possible to make in-sample predictions")
    @pytest.mark.parametrize(
        "model, transforms",
        [
            (BATSModel(use_trend=True), []),
            (TBATSModel(use_trend=True), []),
            (
                DeepARModel(trainer_params=dict(max_epochs=1), lr=0.01, decoder_length=1, encoder_length=1),
                [],
            ),
            (
                TFTModel(
                    trainer_params=dict(max_epochs=1),
                    lr=0.01,
                    dataset_builder=PytorchForecastingDatasetBuilder(
                        max_encoder_length=21,
                        min_encoder_length=21,
                        max_prediction_length=5,
                        time_varying_known_reals=["time_idx"],
                        time_varying_unknown_reals=["target"],
                        static_categoricals=["segment"],
                        target_normalizer=None,
                    ),
                ),
                [],
            ),
        ],
    )
    def test_forecast_in_sample_full_not_implemented(self, model, transforms, example_tsds):
        _test_prediction_in_sample_full(example_tsds, model, transforms, method_name="forecast")


class TestForecastInSampleSuffixNoTarget:
    """Test forecast on suffix of train dataset with filling target with NaNs.

    Expected that NaNs are filled after prediction.
    """

    @staticmethod
    def _test_forecast_in_sample_suffix_no_target(ts, model, transforms, num_skip_points):
        df = ts.to_pandas()

        # fitting
        ts.fit_transform(transforms)
        model.fit(ts)

        # forecasting
        forecast_ts = TSDataset(df, freq="D")
        forecast_ts.transform(ts.transforms)
        forecast_ts.df.loc[forecast_ts.index[num_skip_points] :, pd.IndexSlice[:, "target"]] = np.NaN
        prediction_size = len(forecast_ts.index) - num_skip_points
        forecast_ts.df = forecast_ts.df.iloc[(num_skip_points - model.context_size) :]
        forecast_ts = make_forecast(model=model, ts=forecast_ts, prediction_size=prediction_size)

        # checking
        forecast_df = forecast_ts.to_pandas(flatten=True)
        assert not np.any(forecast_df["target"].isna())

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
            (RNNModel(input_size=1, encoder_length=7, decoder_length=7, trainer_params=dict(max_epochs=1)), []),
        ],
    )
    def test_forecast_in_sample_suffix_no_target(self, model, transforms, example_tsds):
        self._test_forecast_in_sample_suffix_no_target(example_tsds, model, transforms, num_skip_points=50)

    @to_be_fixed(raises=NotImplementedError, match="It is not possible to make in-sample predictions")
    @pytest.mark.parametrize(
        "model, transforms",
        [
            (BATSModel(use_trend=True), []),
            (TBATSModel(use_trend=True), []),
            (
                DeepARModel(trainer_params=dict(max_epochs=1), lr=0.01, decoder_length=1, encoder_length=1),
                [],
            ),
            (
                TFTModel(
                    trainer_params=dict(max_epochs=1),
                    lr=0.01,
                    dataset_builder=PytorchForecastingDatasetBuilder(
                        max_encoder_length=21,
                        min_encoder_length=21,
                        max_prediction_length=5,
                        time_varying_known_reals=["time_idx"],
                        time_varying_unknown_reals=["target"],
                        static_categoricals=["segment"],
                        target_normalizer=None,
                    ),
                ),
                [],
            ),
        ],
    )
    def test_forecast_in_sample_suffix_no_target_failed_not_implemented_in_sample(
        self, model, transforms, example_tsds
    ):
        self._test_forecast_in_sample_suffix_no_target(example_tsds, model, transforms, num_skip_points=50)


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
            (RNNModel(input_size=1, encoder_length=7, decoder_length=7, trainer_params=dict(max_epochs=1)), []),
        ],
    )
    def test_forecast_in_sample_suffix(self, model, transforms, example_tsds):
        _test_prediction_in_sample_suffix(example_tsds, model, transforms, method_name="forecast", num_skip_points=50)

    @to_be_fixed(raises=NotImplementedError, match="It is not possible to make in-sample predictions")
    @pytest.mark.parametrize(
        "model, transforms",
        [
            (BATSModel(use_trend=True), []),
            (TBATSModel(use_trend=True), []),
            (
                DeepARModel(trainer_params=dict(max_epochs=1), lr=0.01, decoder_length=1, encoder_length=1),
                [],
            ),
            (
                TFTModel(
                    trainer_params=dict(max_epochs=1),
                    lr=0.01,
                    dataset_builder=PytorchForecastingDatasetBuilder(
                        max_encoder_length=21,
                        min_encoder_length=21,
                        max_prediction_length=5,
                        time_varying_known_reals=["time_idx"],
                        time_varying_unknown_reals=["target"],
                        static_categoricals=["segment"],
                        target_normalizer=None,
                    ),
                ),
                [],
            ),
        ],
    )
    def test_forecast_in_sample_suffix_failed_not_implemented_in_sample(self, model, transforms, example_tsds):
        _test_prediction_in_sample_suffix(example_tsds, model, transforms, method_name="forecast", num_skip_points=50)


class TestForecastOutSamplePrefix:
    """Test forecast on prefix of future dataset.

    Expected that predictions on prefix match prefix of predictions on full future dataset.
    """

    @staticmethod
    def _test_forecast_out_sample_prefix(ts, model, transforms, full_prediction_size=5, prefix_prediction_size=3):
        prediction_size_diff = full_prediction_size - prefix_prediction_size

        # fitting
        ts.fit_transform(transforms)
        model.fit(ts)

        # forecasting full
        import torch  # TODO: remove after fix at issue-802

        torch.manual_seed(11)

        forecast_full_ts = ts.make_future(future_steps=full_prediction_size, tail_steps=model.context_size)
        forecast_full_ts = make_forecast(model=model, ts=forecast_full_ts, prediction_size=full_prediction_size)

        # forecasting only prefix
        torch.manual_seed(11)  # TODO: remove after fix at issue-802

        forecast_prefix_ts = ts.make_future(future_steps=full_prediction_size, tail_steps=model.context_size)
        forecast_prefix_ts.df = forecast_prefix_ts.df.iloc[:-prediction_size_diff]
        forecast_prefix_ts = make_forecast(model=model, ts=forecast_prefix_ts, prediction_size=prefix_prediction_size)

        # checking
        forecast_full_df = forecast_full_ts.to_pandas()
        forecast_prefix_df = forecast_prefix_ts.to_pandas()
        assert_frame_equal(forecast_prefix_df, forecast_full_df.iloc[:prefix_prediction_size])

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
                DeepARModel(trainer_params=dict(max_epochs=5), lr=0.01, decoder_length=5, encoder_length=5),
                [],
            ),
            (
                TFTModel(
                    trainer_params=dict(max_epochs=1),
                    lr=0.01,
                    dataset_builder=PytorchForecastingDatasetBuilder(
                        max_encoder_length=21,
                        min_encoder_length=21,
                        max_prediction_length=5,
                        time_varying_known_reals=["time_idx"],
                        time_varying_unknown_reals=["target"],
                        static_categoricals=["segment"],
                        target_normalizer=None,
                    ),
                ),
                [],
            ),
            (RNNModel(input_size=1, encoder_length=7, decoder_length=7, trainer_params=dict(max_epochs=1)), []),
        ],
    )
    def test_forecast_out_sample_prefix(self, model, transforms, example_tsds):
        self._test_forecast_out_sample_prefix(example_tsds, model, transforms)


class TestForecastOutSampleSuffix:
    """Test forecast on suffix of future dataset.

    Expected that predictions on suffix match suffix of predictions on full future dataset.
    """

    @staticmethod
    def _test_forecast_out_sample_suffix(ts, model, transforms, full_prediction_size=5, suffix_prediction_size=3):
        prediction_size_diff = full_prediction_size - suffix_prediction_size

        # fitting
        ts.fit_transform(transforms)
        model.fit(ts)

        # forecasting full
        forecast_full_ts = ts.make_future(future_steps=full_prediction_size, tail_steps=model.context_size)
        forecast_full_ts = make_forecast(model=model, ts=forecast_full_ts, prediction_size=full_prediction_size)

        # forecasting only suffix
        forecast_gap_ts = ts.make_future(future_steps=full_prediction_size, tail_steps=model.context_size)
        if isinstance(model, get_args(ContextRequiredModelType)):
            # firstly we should forecast prefix to use it as a context
            forecast_prefix_ts = deepcopy(forecast_gap_ts)
            forecast_prefix_ts.df = forecast_prefix_ts.df.iloc[:-suffix_prediction_size]
            model.forecast(forecast_prefix_ts, prediction_size=prediction_size_diff)
            forecast_gap_ts.df = forecast_gap_ts.df.combine_first(forecast_prefix_ts.df)

            # forecast suffix with known context for it
            forecast_gap_ts = model.forecast(forecast_gap_ts, prediction_size=suffix_prediction_size)
        else:
            forecast_gap_ts.df = forecast_gap_ts.df.iloc[prediction_size_diff:]
            forecast_gap_ts = model.forecast(forecast_gap_ts)

        # checking
        forecast_full_df = forecast_full_ts.to_pandas()
        forecast_gap_df = forecast_gap_ts.to_pandas()
        assert_frame_equal(forecast_gap_df, forecast_full_df.iloc[prediction_size_diff:])

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
        self._test_forecast_out_sample_suffix(example_tsds, model, transforms)

    @to_be_fixed(raises=AssertionError)
    @pytest.mark.parametrize(
        "model, transforms",
        [
            (RNNModel(input_size=1, encoder_length=7, decoder_length=7, trainer_params=dict(max_epochs=1)), []),
        ],
    )
    def test_forecast_out_sample_suffix_failed(self, model, transforms, example_tsds):
        self._test_forecast_out_sample_suffix(example_tsds, model, transforms)

    @to_be_fixed(raises=NotImplementedError, match="You can only forecast from the next point after the last one")
    @pytest.mark.parametrize(
        "model, transforms",
        [
            (
                DeepARModel(trainer_params=dict(max_epochs=5), lr=0.01, decoder_length=5, encoder_length=5),
                [],
            ),
            (
                TFTModel(
                    trainer_params=dict(max_epochs=1),
                    lr=0.01,
                    dataset_builder=PytorchForecastingDatasetBuilder(
                        max_encoder_length=21,
                        min_encoder_length=21,
                        max_prediction_length=5,
                        time_varying_known_reals=["time_idx"],
                        time_varying_unknown_reals=["target"],
                        static_categoricals=["segment"],
                        target_normalizer=None,
                    ),
                ),
                [],
            ),
        ],
    )
    def test_forecast_out_sample_suffix_failed_not_implemented(self, model, transforms, example_tsds):
        self._test_forecast_out_sample_suffix(example_tsds, model, transforms)


class TestForecastMixedInOutSample:
    """Test forecast on mixture of in-sample and out-sample.

    Expected that target values are filled after prediction.
    """

    @staticmethod
    def _test_forecast_mixed_in_out_sample(ts, model, transforms, num_skip_points=50, future_prediction_size=5):
        # fitting
        df = ts.to_pandas()
        ts.fit_transform(transforms)
        model.fit(ts)

        # forecasting mixed in-sample and out-sample
        future_ts = ts.make_future(future_steps=future_prediction_size)
        future_df = future_ts.to_pandas().loc[:, pd.IndexSlice[:, "target"]]
        df_full = pd.concat((df, future_df))
        forecast_full_ts = TSDataset(df=df_full, freq=ts.freq)
        forecast_full_ts.transform(ts.transforms)
        forecast_full_ts.df = forecast_full_ts.df.iloc[(num_skip_points - model.context_size) :]
        full_prediction_size = len(forecast_full_ts.index) - model.context_size
        forecast_full_ts = make_forecast(model=model, ts=forecast_full_ts, prediction_size=full_prediction_size)

        # checking
        forecast_full_df = forecast_full_ts.to_pandas(flatten=True)
        assert not np.any(forecast_full_df["target"].isna())

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
            (RNNModel(input_size=1, encoder_length=7, decoder_length=7, trainer_params=dict(max_epochs=1)), []),
        ],
    )
    def test_forecast_mixed_in_out_sample(self, model, transforms, example_tsds):
        self._test_forecast_mixed_in_out_sample(example_tsds, model, transforms)

    @to_be_fixed(raises=NotImplementedError, match="It is not possible to make in-sample predictions")
    @pytest.mark.parametrize(
        "model, transforms",
        [
            (BATSModel(use_trend=True), []),
            (TBATSModel(use_trend=True), []),
            (
                DeepARModel(trainer_params=dict(max_epochs=5), lr=0.01, decoder_length=5, encoder_length=5),
                [],
            ),
            (
                TFTModel(
                    trainer_params=dict(max_epochs=1),
                    lr=0.01,
                    dataset_builder=PytorchForecastingDatasetBuilder(
                        max_encoder_length=21,
                        min_encoder_length=21,
                        max_prediction_length=5,
                        time_varying_known_reals=["time_idx"],
                        time_varying_unknown_reals=["target"],
                        static_categoricals=["segment"],
                        target_normalizer=None,
                    ),
                ),
                [],
            ),
        ],
    )
    def test_forecast_mixed_in_out_sample_failed_not_implemented_in_sample(self, model, transforms, example_tsds):
        self._test_forecast_mixed_in_out_sample(example_tsds, model, transforms)
