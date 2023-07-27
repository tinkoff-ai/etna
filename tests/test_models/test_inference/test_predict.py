from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from pandas.util.testing import assert_frame_equal
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.data import NaNLabelEncoder

from etna.datasets import TSDataset
from etna.models import AutoARIMAModel
from etna.models import BATSModel
from etna.models import CatBoostMultiSegmentModel
from etna.models import CatBoostPerSegmentModel
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
from etna.models import StatsForecastARIMAModel
from etna.models import StatsForecastAutoARIMAModel
from etna.models import StatsForecastAutoCESModel
from etna.models import StatsForecastAutoETSModel
from etna.models import StatsForecastAutoThetaModel
from etna.models import TBATSModel
from etna.models.nn import DeepARModel
from etna.models.nn import DeepStateModel
from etna.models.nn import MLPModel
from etna.models.nn import NBeatsGenericModel
from etna.models.nn import NBeatsInterpretableModel
from etna.models.nn import PatchTSModel
from etna.models.nn import PytorchForecastingDatasetBuilder
from etna.models.nn import RNNModel
from etna.models.nn import TFTModel
from etna.models.nn.deepstate import CompositeSSM
from etna.models.nn.deepstate import WeeklySeasonalitySSM
from etna.transforms import LagTransform
from etna.transforms import SegmentEncoderTransform
from tests.test_models.test_inference.common import _test_prediction_in_sample_full
from tests.test_models.test_inference.common import _test_prediction_in_sample_suffix
from tests.test_models.test_inference.common import make_prediction
from tests.utils import select_segments_subset
from tests.utils import to_be_fixed


def make_predict(model, ts, prediction_size) -> TSDataset:
    return make_prediction(model=model, ts=ts, prediction_size=prediction_size, method_name="predict")


class TestPredictInSampleFull:
    """Test predict on full train dataset.

    Expected that there are no NaNs after prediction and targets are changed compared to original.
    """

    @pytest.mark.parametrize(
        "model, transforms",
        [
            (CatBoostMultiSegmentModel(), [LagTransform(in_column="target", lags=[2, 3])]),
            (CatBoostPerSegmentModel(), [LagTransform(in_column="target", lags=[2, 3])]),
            (ProphetModel(), []),
            (SARIMAXModel(), []),
            (AutoARIMAModel(), []),
            (HoltModel(), []),
            (HoltWintersModel(), []),
            (SimpleExpSmoothingModel(), []),
            (BATSModel(use_trend=True), []),
            (TBATSModel(use_trend=True), []),
            (StatsForecastARIMAModel(), []),
            (StatsForecastAutoARIMAModel(), []),
            (StatsForecastAutoCESModel(), []),
            (StatsForecastAutoETSModel(), []),
            (StatsForecastAutoThetaModel(), []),
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
    def test_predict_in_sample_full_failed_nans_sklearn(self, model, transforms, example_tsds):
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
            (
                DeepARModel(
                    dataset_builder=PytorchForecastingDatasetBuilder(
                        max_encoder_length=1,
                        max_prediction_length=1,
                        time_varying_known_reals=["time_idx"],
                        time_varying_unknown_reals=["target"],
                        target_normalizer=GroupNormalizer(groups=["segment"]),
                    ),
                    trainer_params=dict(max_epochs=1),
                    lr=0.01,
                ),
                [],
            ),
            (
                TFTModel(
                    dataset_builder=PytorchForecastingDatasetBuilder(
                        max_encoder_length=21,
                        min_encoder_length=21,
                        max_prediction_length=5,
                        time_varying_known_reals=["time_idx"],
                        time_varying_unknown_reals=["target"],
                        static_categoricals=["segment"],
                        target_normalizer=None,
                    ),
                    trainer_params=dict(max_epochs=1),
                    lr=0.01,
                ),
                [],
            ),
            (RNNModel(input_size=1, encoder_length=7, decoder_length=7, trainer_params=dict(max_epochs=1)), []),
            (PatchTSModel(encoder_length=7, decoder_length=7, trainer_params=dict(max_epochs=1)), []),
            (
                MLPModel(input_size=2, hidden_size=[10], decoder_length=7, trainer_params=dict(max_epochs=1)),
                [LagTransform(in_column="target", lags=[2, 3])],
            ),
            (
                DeepStateModel(
                    ssm=CompositeSSM(seasonal_ssms=[WeeklySeasonalitySSM()]),
                    input_size=1,
                    encoder_length=7,
                    decoder_length=7,
                    trainer_params=dict(max_epochs=1),
                ),
                [SegmentEncoderTransform()],
            ),
            (NBeatsInterpretableModel(input_size=7, output_size=7, trainer_params=dict(max_epochs=1)), []),
            (NBeatsGenericModel(input_size=7, output_size=7, trainer_params=dict(max_epochs=1)), []),
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

    Expected that there are no NaNs after prediction and targets are changed compared to original.
    """

    @pytest.mark.parametrize(
        "model, transforms",
        [
            (CatBoostPerSegmentModel(), [LagTransform(in_column="target", lags=[2, 3])]),
            (CatBoostMultiSegmentModel(), [LagTransform(in_column="target", lags=[2, 3])]),
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
            (BATSModel(use_trend=True), []),
            (TBATSModel(use_trend=True), []),
            (StatsForecastARIMAModel(), []),
            (StatsForecastAutoARIMAModel(), []),
            (StatsForecastAutoCESModel(), []),
            (StatsForecastAutoETSModel(), []),
            (StatsForecastAutoThetaModel(), []),
        ],
    )
    def test_predict_in_sample_suffix(self, model, transforms, example_tsds):
        _test_prediction_in_sample_suffix(example_tsds, model, transforms, method_name="predict", num_skip_points=50)

    @to_be_fixed(raises=NotImplementedError, match="Method predict isn't currently implemented")
    @pytest.mark.parametrize(
        "model, transforms",
        [
            (
                DeepARModel(
                    dataset_builder=PytorchForecastingDatasetBuilder(
                        max_encoder_length=1,
                        max_prediction_length=1,
                        time_varying_known_reals=["time_idx"],
                        time_varying_unknown_reals=["target"],
                        target_normalizer=GroupNormalizer(groups=["segment"]),
                    ),
                    trainer_params=dict(max_epochs=1),
                    lr=0.01,
                ),
                [],
            ),
            (
                TFTModel(
                    dataset_builder=PytorchForecastingDatasetBuilder(
                        max_encoder_length=21,
                        min_encoder_length=21,
                        max_prediction_length=5,
                        time_varying_known_reals=["time_idx"],
                        time_varying_unknown_reals=["target"],
                        static_categoricals=["segment"],
                        target_normalizer=None,
                    ),
                    trainer_params=dict(max_epochs=1),
                    lr=0.01,
                ),
                [],
            ),
            (RNNModel(input_size=1, encoder_length=7, decoder_length=7, trainer_params=dict(max_epochs=1)), []),
            (PatchTSModel(encoder_length=7, decoder_length=7, trainer_params=dict(max_epochs=1)), []),
            (
                MLPModel(input_size=2, hidden_size=[10], decoder_length=7, trainer_params=dict(max_epochs=1)),
                [LagTransform(in_column="target", lags=[2, 3])],
            ),
            (
                DeepStateModel(
                    ssm=CompositeSSM(seasonal_ssms=[WeeklySeasonalitySSM()]),
                    input_size=1,
                    encoder_length=7,
                    decoder_length=7,
                    trainer_params=dict(max_epochs=1),
                ),
                [SegmentEncoderTransform()],
            ),
            (NBeatsInterpretableModel(input_size=7, output_size=7, trainer_params=dict(max_epochs=1)), []),
            (NBeatsGenericModel(input_size=7, output_size=7, trainer_params=dict(max_epochs=1)), []),
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

    Expected that there are no NaNs after prediction and targets are changed compared to original.
    """

    @staticmethod
    def _test_predict_out_sample(ts, model, transforms, prediction_size=5):
        train_ts, _ = ts.train_test_split(test_size=prediction_size)
        forecast_ts = TSDataset(df=ts.df, freq=ts.freq)
        df = forecast_ts.to_pandas()

        # fitting
        train_ts.fit_transform(transforms)
        model.fit(train_ts)

        # forecasting
        forecast_ts.transform(transforms)
        to_remain = model.context_size + prediction_size
        forecast_ts.df = forecast_ts.df.iloc[-to_remain:]
        forecast_ts = make_predict(model=model, ts=forecast_ts, prediction_size=prediction_size)

        # checking
        forecast_df = forecast_ts.to_pandas(flatten=True)
        assert not np.any(forecast_df["target"].isna())
        original_target = TSDataset.to_flatten(df.iloc[-to_remain:])["target"]
        assert not forecast_df["target"].equals(original_target)

    @pytest.mark.parametrize(
        "model, transforms",
        [
            (CatBoostPerSegmentModel(), [LagTransform(in_column="target", lags=[5, 6])]),
            (CatBoostMultiSegmentModel(), [LagTransform(in_column="target", lags=[5, 6])]),
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
            (
                DeepARModel(
                    dataset_builder=PytorchForecastingDatasetBuilder(
                        max_encoder_length=5,
                        max_prediction_length=5,
                        time_varying_known_reals=["time_idx"],
                        time_varying_unknown_reals=["target"],
                        target_normalizer=GroupNormalizer(groups=["segment"]),
                    ),
                    trainer_params=dict(max_epochs=1),
                    lr=0.01,
                ),
                [],
            ),
            (
                TFTModel(
                    dataset_builder=PytorchForecastingDatasetBuilder(
                        max_encoder_length=21,
                        min_encoder_length=21,
                        max_prediction_length=5,
                        time_varying_known_reals=["time_idx"],
                        time_varying_unknown_reals=["target"],
                        static_categoricals=["segment"],
                        target_normalizer=None,
                    ),
                    trainer_params=dict(max_epochs=1),
                    lr=0.01,
                ),
                [],
            ),
            (RNNModel(input_size=1, encoder_length=7, decoder_length=7, trainer_params=dict(max_epochs=1)), []),
            (PatchTSModel(encoder_length=7, decoder_length=7, trainer_params=dict(max_epochs=1)), []),
            (
                MLPModel(input_size=2, hidden_size=[10], decoder_length=7, trainer_params=dict(max_epochs=1)),
                [LagTransform(in_column="target", lags=[5, 6])],
            ),
            (
                DeepStateModel(
                    ssm=CompositeSSM(seasonal_ssms=[WeeklySeasonalitySSM()]),
                    input_size=1,
                    encoder_length=7,
                    decoder_length=7,
                    trainer_params=dict(max_epochs=1),
                ),
                [SegmentEncoderTransform()],
            ),
            (NBeatsInterpretableModel(input_size=7, output_size=7, trainer_params=dict(max_epochs=1)), []),
            (NBeatsGenericModel(input_size=7, output_size=7, trainer_params=dict(max_epochs=1)), []),
        ],
    )
    def test_predict_out_sample_failed_not_implemented_predict(self, model, transforms, example_tsds):
        self._test_predict_out_sample(example_tsds, model, transforms)

    @to_be_fixed(raises=NotImplementedError, match="This model can't make predict on future out-of-sample data")
    @pytest.mark.parametrize(
        "model, transforms",
        [
            (BATSModel(use_trend=True), []),
            (TBATSModel(use_trend=True), []),
            (StatsForecastARIMAModel(), []),
            (StatsForecastAutoARIMAModel(), []),
            (StatsForecastAutoCESModel(), []),
            (StatsForecastAutoETSModel(), []),
            (StatsForecastAutoThetaModel(), []),
        ],
    )
    def test_predict_out_sample_failed_not_implemented_out_sample(self, model, transforms, example_tsds):
        self._test_predict_out_sample(example_tsds, model, transforms)


class TestPredictOutSamplePrefix:
    """Test predict on prefix of future dataset.

    Expected that predictions on prefix match prefix of predictions on full future dataset.
    """

    @staticmethod
    def _test_predict_out_sample_prefix(ts, model, transforms, full_prediction_size=5, prefix_prediction_size=3):
        prediction_size_diff = full_prediction_size - prefix_prediction_size
        train_ts, _ = ts.train_test_split(test_size=full_prediction_size)
        forecast_full_ts = TSDataset(df=ts.df, freq=ts.freq)
        forecast_prefix_ts = TSDataset(df=ts.df, freq=ts.freq)

        # fitting
        train_ts.fit_transform(transforms)
        model.fit(train_ts)

        # forecasting full
        forecast_full_ts.transform(transforms)
        to_remain = model.context_size + full_prediction_size
        forecast_full_ts.df = forecast_full_ts.df.iloc[-to_remain:]
        forecast_full_ts = make_predict(model=model, ts=forecast_full_ts, prediction_size=full_prediction_size)

        # forecasting only prefix
        forecast_prefix_ts.transform(transforms)
        forecast_prefix_ts.df = forecast_prefix_ts.df.iloc[-to_remain:-prediction_size_diff]
        forecast_prefix_ts = make_predict(model=model, ts=forecast_prefix_ts, prediction_size=prefix_prediction_size)

        # checking
        forecast_full_df = forecast_full_ts.to_pandas()
        forecast_prefix_df = forecast_prefix_ts.to_pandas()
        assert_frame_equal(forecast_prefix_df, forecast_full_df.iloc[:prefix_prediction_size])

    @pytest.mark.parametrize(
        "model, transforms",
        [
            (CatBoostPerSegmentModel(), [LagTransform(in_column="target", lags=[5, 6])]),
            (CatBoostMultiSegmentModel(), [LagTransform(in_column="target", lags=[5, 6])]),
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
    def test_predict_out_sample_prefix(self, model, transforms, example_tsds):
        self._test_predict_out_sample_prefix(example_tsds, model, transforms)

    @to_be_fixed(raises=NotImplementedError, match="Method predict isn't currently implemented")
    @pytest.mark.parametrize(
        "model, transforms",
        [
            (
                DeepARModel(
                    dataset_builder=PytorchForecastingDatasetBuilder(
                        max_encoder_length=5,
                        max_prediction_length=5,
                        time_varying_known_reals=["time_idx"],
                        time_varying_unknown_reals=["target"],
                        target_normalizer=GroupNormalizer(groups=["segment"]),
                    ),
                    trainer_params=dict(max_epochs=1),
                    lr=0.01,
                ),
                [],
            ),
            (
                TFTModel(
                    dataset_builder=PytorchForecastingDatasetBuilder(
                        max_encoder_length=21,
                        min_encoder_length=21,
                        max_prediction_length=5,
                        time_varying_known_reals=["time_idx"],
                        time_varying_unknown_reals=["target"],
                        static_categoricals=["segment"],
                        target_normalizer=None,
                    ),
                    trainer_params=dict(max_epochs=1),
                    lr=0.01,
                ),
                [],
            ),
            (RNNModel(input_size=1, encoder_length=7, decoder_length=7, trainer_params=dict(max_epochs=1)), []),
            (PatchTSModel(encoder_length=7, decoder_length=7, trainer_params=dict(max_epochs=1)), []),
            (
                MLPModel(input_size=2, hidden_size=[10], decoder_length=7, trainer_params=dict(max_epochs=1)),
                [LagTransform(in_column="target", lags=[5, 6])],
            ),
            (
                DeepStateModel(
                    ssm=CompositeSSM(seasonal_ssms=[WeeklySeasonalitySSM()]),
                    input_size=1,
                    encoder_length=7,
                    decoder_length=7,
                    trainer_params=dict(max_epochs=1),
                ),
                [SegmentEncoderTransform()],
            ),
            (NBeatsInterpretableModel(input_size=7, output_size=7, trainer_params=dict(max_epochs=1)), []),
            (NBeatsGenericModel(input_size=7, output_size=7, trainer_params=dict(max_epochs=1)), []),
        ],
    )
    def test_predict_out_sample_prefix_failed_not_implemented_predict(self, model, transforms, example_tsds):
        self._test_predict_out_sample_prefix(example_tsds, model, transforms)

    @to_be_fixed(raises=NotImplementedError, match="This model can't make predict on future out-of-sample data")
    @pytest.mark.parametrize(
        "model, transforms",
        [
            (BATSModel(use_trend=True), []),
            (TBATSModel(use_trend=True), []),
            (StatsForecastARIMAModel(), []),
            (StatsForecastAutoARIMAModel(), []),
            (StatsForecastAutoCESModel(), []),
            (StatsForecastAutoETSModel(), []),
            (StatsForecastAutoThetaModel(), []),
        ],
    )
    def test_predict_out_sample_prefix_failed_not_implemented_out_sample(self, model, transforms, example_tsds):
        self._test_predict_out_sample_prefix(example_tsds, model, transforms)


class TestPredictOutSampleSuffix:
    """Test predict on suffix of future dataset.

    Expected that predictions on suffix match suffix of predictions on full future dataset.
    """

    @staticmethod
    def _test_predict_out_sample_suffix(ts, model, transforms, full_prediction_size=5, suffix_prediction_size=3):
        prediction_size_diff = full_prediction_size - suffix_prediction_size
        train_ts, _ = ts.train_test_split(test_size=full_prediction_size)
        forecast_full_ts = TSDataset(df=ts.df, freq=ts.freq)
        forecast_suffix_ts = TSDataset(df=ts.df, freq=ts.freq)

        # fitting
        train_ts.fit_transform(transforms)
        model.fit(train_ts)

        # forecasting full
        forecast_full_ts.transform(transforms)
        to_remain = model.context_size + full_prediction_size
        forecast_full_ts.df = forecast_full_ts.df.iloc[-to_remain:]
        forecast_full_ts = make_predict(model=model, ts=forecast_full_ts, prediction_size=full_prediction_size)

        # forecasting only suffix
        forecast_suffix_ts.transform(transforms)
        to_remain = model.context_size + suffix_prediction_size
        forecast_suffix_ts.df = forecast_suffix_ts.df.iloc[-to_remain:]
        forecast_suffix_ts = make_predict(model=model, ts=forecast_suffix_ts, prediction_size=suffix_prediction_size)

        # checking
        forecast_full_df = forecast_full_ts.to_pandas()
        forecast_suffix_df = forecast_suffix_ts.to_pandas()
        assert_frame_equal(forecast_suffix_df, forecast_full_df.iloc[prediction_size_diff:])

    @pytest.mark.parametrize(
        "model, transforms",
        [
            (CatBoostPerSegmentModel(), [LagTransform(in_column="target", lags=[5, 6])]),
            (CatBoostMultiSegmentModel(), [LagTransform(in_column="target", lags=[5, 6])]),
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
    def test_predict_out_sample_suffix(self, model, transforms, example_tsds):
        self._test_predict_out_sample_suffix(example_tsds, model, transforms)

    @to_be_fixed(raises=NotImplementedError, match="Method predict isn't currently implemented")
    @pytest.mark.parametrize(
        "model, transforms",
        [
            (
                DeepARModel(
                    dataset_builder=PytorchForecastingDatasetBuilder(
                        max_encoder_length=5,
                        max_prediction_length=5,
                        time_varying_known_reals=["time_idx"],
                        time_varying_unknown_reals=["target"],
                        target_normalizer=GroupNormalizer(groups=["segment"]),
                    ),
                    trainer_params=dict(max_epochs=1),
                    lr=0.01,
                ),
                [],
            ),
            (
                TFTModel(
                    dataset_builder=PytorchForecastingDatasetBuilder(
                        max_encoder_length=21,
                        min_encoder_length=21,
                        max_prediction_length=5,
                        time_varying_known_reals=["time_idx"],
                        time_varying_unknown_reals=["target"],
                        static_categoricals=["segment"],
                        target_normalizer=None,
                    ),
                    trainer_params=dict(max_epochs=1),
                    lr=0.01,
                ),
                [],
            ),
            (RNNModel(input_size=1, encoder_length=7, decoder_length=7, trainer_params=dict(max_epochs=1)), []),
            (PatchTSModel(encoder_length=7, decoder_length=7, trainer_params=dict(max_epochs=1)), []),
            (
                MLPModel(input_size=2, hidden_size=[10], decoder_length=7, trainer_params=dict(max_epochs=1)),
                [LagTransform(in_column="target", lags=[5, 6])],
            ),
            (
                DeepStateModel(
                    ssm=CompositeSSM(seasonal_ssms=[WeeklySeasonalitySSM()]),
                    input_size=1,
                    encoder_length=7,
                    decoder_length=7,
                    trainer_params=dict(max_epochs=1),
                ),
                [SegmentEncoderTransform()],
            ),
            (
                DeepStateModel(
                    ssm=CompositeSSM(seasonal_ssms=[WeeklySeasonalitySSM()]),
                    input_size=1,
                    encoder_length=7,
                    decoder_length=7,
                    trainer_params=dict(max_epochs=1),
                ),
                [SegmentEncoderTransform()],
            ),
            (NBeatsInterpretableModel(input_size=7, output_size=7, trainer_params=dict(max_epochs=1)), []),
            (NBeatsGenericModel(input_size=7, output_size=7, trainer_params=dict(max_epochs=1)), []),
        ],
    )
    def test_predict_out_sample_suffix_failed_not_implemented_predict(self, model, transforms, example_tsds):
        self._test_predict_out_sample_suffix(example_tsds, model, transforms)

    @to_be_fixed(raises=NotImplementedError, match="This model can't make predict on future out-of-sample data")
    @pytest.mark.parametrize(
        "model, transforms",
        [
            (BATSModel(use_trend=True), []),
            (TBATSModel(use_trend=True), []),
            (StatsForecastARIMAModel(), []),
            (StatsForecastAutoARIMAModel(), []),
            (StatsForecastAutoCESModel(), []),
            (StatsForecastAutoETSModel(), []),
            (StatsForecastAutoThetaModel(), []),
        ],
    )
    def test_predict_out_sample_suffix_failed_not_implemented_out_sample(self, model, transforms, example_tsds):
        self._test_predict_out_sample_suffix(example_tsds, model, transforms)


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
        forecast_full_ts.transform(transforms)
        forecast_full_ts.df = forecast_full_ts.df.iloc[(num_skip_points - model.context_size) :]
        full_prediction_size = len(forecast_full_ts.index) - model.context_size
        forecast_full_ts = make_predict(model=model, ts=forecast_full_ts, prediction_size=full_prediction_size)

        # predicting only in sample
        forecast_in_sample_ts = TSDataset(train_df, freq=ts.freq)
        forecast_in_sample_ts.transform(transforms)
        to_skip = num_skip_points - model.context_size
        forecast_in_sample_ts.df = forecast_in_sample_ts.df.iloc[to_skip:]
        in_sample_prediction_size = len(forecast_in_sample_ts.index) - model.context_size
        forecast_in_sample_ts = make_predict(
            model=model, ts=forecast_in_sample_ts, prediction_size=in_sample_prediction_size
        )

        # predicting only out sample
        forecast_out_sample_ts = TSDataset(df=df_full, freq=ts.freq)
        forecast_out_sample_ts.transform(transforms)
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
            (CatBoostPerSegmentModel(), [LagTransform(in_column="target", lags=[5, 6])]),
            (CatBoostMultiSegmentModel(), [LagTransform(in_column="target", lags=[5, 6])]),
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
            (
                DeepARModel(
                    dataset_builder=PytorchForecastingDatasetBuilder(
                        max_encoder_length=5,
                        max_prediction_length=5,
                        time_varying_known_reals=["time_idx"],
                        time_varying_unknown_reals=["target"],
                        target_normalizer=GroupNormalizer(groups=["segment"]),
                    ),
                    trainer_params=dict(max_epochs=1),
                    lr=0.01,
                ),
                [],
            ),
            (
                TFTModel(
                    dataset_builder=PytorchForecastingDatasetBuilder(
                        max_encoder_length=21,
                        min_encoder_length=21,
                        max_prediction_length=5,
                        time_varying_known_reals=["time_idx"],
                        time_varying_unknown_reals=["target"],
                        static_categoricals=["segment"],
                        target_normalizer=None,
                    ),
                    trainer_params=dict(max_epochs=1),
                    lr=0.01,
                ),
                [],
            ),
            (RNNModel(input_size=1, encoder_length=7, decoder_length=7, trainer_params=dict(max_epochs=1)), []),
            (PatchTSModel(encoder_length=7, decoder_length=7, trainer_params=dict(max_epochs=1)), []),
            (
                MLPModel(input_size=2, hidden_size=[10], decoder_length=7, trainer_params=dict(max_epochs=1)),
                [LagTransform(in_column="target", lags=[5, 6])],
            ),
            (
                DeepStateModel(
                    ssm=CompositeSSM(seasonal_ssms=[WeeklySeasonalitySSM()]),
                    input_size=1,
                    encoder_length=7,
                    decoder_length=7,
                    trainer_params=dict(max_epochs=1),
                ),
                [SegmentEncoderTransform()],
            ),
            (NBeatsInterpretableModel(input_size=7, output_size=7, trainer_params=dict(max_epochs=1)), []),
            (NBeatsGenericModel(input_size=7, output_size=7, trainer_params=dict(max_epochs=1)), []),
        ],
    )
    def test_predict_mixed_in_out_sample_failed_not_implemented_predict(self, model, transforms, example_tsds):
        self._test_predict_mixed_in_out_sample(example_tsds, model, transforms)

    @to_be_fixed(raises=NotImplementedError, match="This model can't make predict on future out-of-sample data")
    @pytest.mark.parametrize(
        "model, transforms",
        [
            (BATSModel(use_trend=True), []),
            (TBATSModel(use_trend=True), []),
            (StatsForecastARIMAModel(), []),
            (StatsForecastAutoARIMAModel(), []),
            (StatsForecastAutoCESModel(), []),
            (StatsForecastAutoETSModel(), []),
            (StatsForecastAutoThetaModel(), []),
        ],
    )
    def test_predict_mixed_in_out_sample_failed_not_implemented_out_sample(self, model, transforms, example_tsds):
        self._test_predict_mixed_in_out_sample(example_tsds, model, transforms)


class TestPredictSubsetSegments:
    """Test predict on subset of segments on suffix of train dataset.

    Expected that predictions on subset of segments match subset of predictions on full dataset.
    """

    def _test_predict_subset_segments(self, ts, model, transforms, segments, num_skip_points=50):
        prediction_size = len(ts.index) - num_skip_points

        # select subset of tsdataset
        segments = list(set(segments))
        subset_ts = select_segments_subset(ts=deepcopy(ts), segments=segments)

        # fitting
        ts.fit_transform(transforms)
        subset_ts.transform(transforms)
        model.fit(ts)

        # forecasting full
        ts.df = ts.df.iloc[(num_skip_points - model.context_size) :]
        forecast_full_ts = make_predict(model=model, ts=ts, prediction_size=prediction_size)

        # forecasting subset of segments
        subset_ts.df = subset_ts.df.iloc[(num_skip_points - model.context_size) :]
        forecast_subset_ts = make_predict(model=model, ts=subset_ts, prediction_size=prediction_size)

        # checking
        forecast_full_df = forecast_full_ts.to_pandas()
        forecast_subset_df = forecast_subset_ts.to_pandas()
        assert_frame_equal(forecast_subset_df, forecast_full_df.loc[:, pd.IndexSlice[segments, :]])

    @pytest.mark.parametrize(
        "model, transforms",
        [
            (CatBoostPerSegmentModel(), [LagTransform(in_column="target", lags=[5, 6])]),
            (CatBoostMultiSegmentModel(), [LagTransform(in_column="target", lags=[5, 6])]),
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
            (StatsForecastARIMAModel(), []),
            (StatsForecastAutoARIMAModel(), []),
            (StatsForecastAutoCESModel(), []),
            (StatsForecastAutoETSModel(), []),
            (StatsForecastAutoThetaModel(), []),
        ],
    )
    def test_predict_subset_segments(self, model, transforms, example_tsds):
        self._test_predict_subset_segments(example_tsds, model, transforms, segments=["segment_2"])

    @to_be_fixed(raises=NotImplementedError, match="Method predict isn't currently implemented")
    @pytest.mark.parametrize(
        "model, transforms",
        [
            (
                DeepARModel(
                    dataset_builder=PytorchForecastingDatasetBuilder(
                        max_encoder_length=5,
                        max_prediction_length=5,
                        time_varying_known_reals=["time_idx"],
                        time_varying_unknown_reals=["target"],
                        target_normalizer=GroupNormalizer(groups=["segment"]),
                    ),
                    trainer_params=dict(max_epochs=1),
                    lr=0.01,
                ),
                [],
            ),
            (
                TFTModel(
                    dataset_builder=PytorchForecastingDatasetBuilder(
                        max_encoder_length=21,
                        min_encoder_length=21,
                        max_prediction_length=5,
                        time_varying_known_reals=["time_idx"],
                        time_varying_unknown_reals=["target"],
                        static_categoricals=["segment"],
                        target_normalizer=None,
                    ),
                    trainer_params=dict(max_epochs=1),
                    lr=0.01,
                ),
                [],
            ),
            (RNNModel(input_size=1, encoder_length=7, decoder_length=7, trainer_params=dict(max_epochs=1)), []),
            (PatchTSModel(encoder_length=7, decoder_length=7, trainer_params=dict(max_epochs=1)), []),
            (
                MLPModel(input_size=2, hidden_size=[10], decoder_length=7, trainer_params=dict(max_epochs=1)),
                [LagTransform(in_column="target", lags=[5, 6])],
            ),
            (
                DeepStateModel(
                    ssm=CompositeSSM(seasonal_ssms=[WeeklySeasonalitySSM()]),
                    input_size=1,
                    encoder_length=7,
                    decoder_length=7,
                    trainer_params=dict(max_epochs=1),
                ),
                [SegmentEncoderTransform()],
            ),
            (NBeatsInterpretableModel(input_size=7, output_size=7, trainer_params=dict(max_epochs=1)), []),
            (NBeatsGenericModel(input_size=7, output_size=7, trainer_params=dict(max_epochs=1)), []),
        ],
    )
    def test_predict_subset_segments_failed_not_implemented_predict(self, model, transforms, example_tsds):
        self._test_predict_subset_segments(example_tsds, model, transforms, segments=["segment_2"])


class TestPredictNewSegments:
    """Test predict on new segments on suffix of train dataset.

    Expected that there are no NaNs after prediction and targets are changed compared to original.
    """

    def _test_predict_new_segments(self, ts, model, transforms, train_segments, num_skip_points=50):
        # create tsdataset with new segments
        train_segments = list(set(train_segments))
        forecast_segments = list(set(ts.segments) - set(train_segments))
        train_ts = select_segments_subset(ts=deepcopy(ts), segments=train_segments)
        test_ts = select_segments_subset(ts=deepcopy(ts), segments=forecast_segments)
        df = test_ts.to_pandas()

        # fitting
        train_ts.fit_transform(transforms)
        test_ts.transform(transforms)
        model.fit(train_ts)

        # forecasting
        test_ts.df = test_ts.df.iloc[(num_skip_points - model.context_size) :]
        prediction_size = len(ts.index) - num_skip_points
        forecast_ts = make_predict(model=model, ts=test_ts, prediction_size=prediction_size)

        # checking
        forecast_df = forecast_ts.to_pandas(flatten=True)
        assert not np.any(forecast_df["target"].isna())
        original_target = TSDataset.to_flatten(df.iloc[(num_skip_points - model.context_size) :])["target"]
        assert not forecast_df["target"].equals(original_target)

    @pytest.mark.parametrize(
        "model, transforms",
        [
            (CatBoostMultiSegmentModel(), [LagTransform(in_column="target", lags=[5, 6])]),
            (LinearMultiSegmentModel(), [LagTransform(in_column="target", lags=[5, 6])]),
            (ElasticMultiSegmentModel(), [LagTransform(in_column="target", lags=[5, 6])]),
            (MovingAverageModel(window=3), []),
            (SeasonalMovingAverageModel(), []),
            (NaiveModel(lag=3), []),
            (DeadlineMovingAverageModel(window=1), []),
        ],
    )
    def test_predict_new_segments(self, model, transforms, example_tsds):
        self._test_predict_new_segments(example_tsds, model, transforms, train_segments=["segment_1"])

    @to_be_fixed(raises=NotImplementedError, match="Method predict isn't currently implemented")
    @pytest.mark.parametrize(
        "model, transforms",
        [
            (
                DeepARModel(
                    dataset_builder=PytorchForecastingDatasetBuilder(
                        max_encoder_length=5,
                        max_prediction_length=5,
                        time_varying_known_reals=["time_idx"],
                        time_varying_unknown_reals=["target"],
                        categorical_encoders={"segment": NaNLabelEncoder(add_nan=True, warn=False)},
                        target_normalizer=GroupNormalizer(groups=["segment"]),
                    ),
                    trainer_params=dict(max_epochs=1),
                    lr=0.01,
                ),
                [],
            ),
            (
                TFTModel(
                    dataset_builder=PytorchForecastingDatasetBuilder(
                        max_encoder_length=21,
                        min_encoder_length=21,
                        max_prediction_length=5,
                        time_varying_known_reals=["time_idx"],
                        time_varying_unknown_reals=["target"],
                        categorical_encoders={"segment": NaNLabelEncoder(add_nan=True, warn=False)},
                        static_categoricals=["segment"],
                        target_normalizer=None,
                    ),
                    trainer_params=dict(max_epochs=1),
                    lr=0.01,
                ),
                [],
            ),
            (RNNModel(input_size=1, encoder_length=7, decoder_length=7, trainer_params=dict(max_epochs=1)), []),
            (PatchTSModel(encoder_length=7, decoder_length=7, trainer_params=dict(max_epochs=1)), []),
            (
                MLPModel(input_size=2, hidden_size=[10], decoder_length=7, trainer_params=dict(max_epochs=1)),
                [LagTransform(in_column="target", lags=[5, 6])],
            ),
            (
                DeepStateModel(
                    ssm=CompositeSSM(seasonal_ssms=[WeeklySeasonalitySSM()]),
                    input_size=0,
                    encoder_length=7,
                    decoder_length=7,
                    trainer_params=dict(max_epochs=1),
                ),
                [],
            ),
            (NBeatsInterpretableModel(input_size=7, output_size=7, trainer_params=dict(max_epochs=1)), []),
            (NBeatsGenericModel(input_size=7, output_size=7, trainer_params=dict(max_epochs=1)), []),
        ],
    )
    def test_predict_new_segments_failed_not_implemented_predict(self, model, transforms, example_tsds):
        self._test_predict_new_segments(example_tsds, model, transforms, train_segments=["segment_1"])

    @pytest.mark.parametrize(
        "model, transforms",
        [
            (CatBoostPerSegmentModel(), [LagTransform(in_column="target", lags=[5, 6])]),
            (LinearPerSegmentModel(), [LagTransform(in_column="target", lags=[5, 6])]),
            (ElasticPerSegmentModel(), [LagTransform(in_column="target", lags=[5, 6])]),
            (AutoARIMAModel(), []),
            (ProphetModel(), []),
            (SARIMAXModel(), []),
            (HoltModel(), []),
            (HoltWintersModel(), []),
            (SimpleExpSmoothingModel(), []),
            (BATSModel(use_trend=True), []),
            (TBATSModel(use_trend=True), []),
            (StatsForecastARIMAModel(), []),
            (StatsForecastAutoARIMAModel(), []),
            (StatsForecastAutoCESModel(), []),
            (StatsForecastAutoETSModel(), []),
            (StatsForecastAutoThetaModel(), []),
        ],
    )
    def test_predict_new_segments_failed_per_segment(self, model, transforms, example_tsds):
        with pytest.raises(NotImplementedError, match="Per-segment models can't make predictions on new segments"):
            self._test_predict_new_segments(example_tsds, model, transforms, train_segments=["segment_1"])
