from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from pandas.util.testing import assert_frame_equal
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.data import NaNLabelEncoder
from typing_extensions import get_args

from etna.datasets import TSDataset
from etna.models import AutoARIMAModel
from etna.models import BATSModel
from etna.models import CatBoostMultiSegmentModel
from etna.models import CatBoostPerSegmentModel
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


def make_forecast(model, ts, prediction_size) -> TSDataset:
    return make_prediction(model=model, ts=ts, prediction_size=prediction_size, method_name="forecast")


class TestForecastInSampleFullNoTarget:
    """Test forecast on full train dataset where target is filled with NaNs.

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
        forecast_ts.transform(transforms)
        forecast_ts.df.loc[:, pd.IndexSlice[:, "target"]] = np.NaN
        prediction_size = len(forecast_ts.index)
        forecast_ts = make_forecast(model=model, ts=forecast_ts, prediction_size=prediction_size)

        # checking
        forecast_df = forecast_ts.to_pandas(flatten=True)
        assert not np.any(forecast_df["target"].isna())

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
        ],
    )
    def test_forecast_in_sample_full_no_target(self, model, transforms, example_tsds):
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
    def test_forecast_in_sample_full_no_target_failed_nans_sklearn(self, model, transforms, example_tsds):
        with pytest.raises(ValueError, match="Input contains NaN, infinity or a value too large"):
            self._test_forecast_in_sample_full_no_target(example_tsds, model, transforms)

    @pytest.mark.parametrize(
        "model, transforms",
        [
            (MovingAverageModel(window=3), []),
            (NaiveModel(lag=3), []),
            (SeasonalMovingAverageModel(), []),
            (DeadlineMovingAverageModel(window=1), []),
            (RNNModel(input_size=1, encoder_length=7, decoder_length=7, trainer_params=dict(max_epochs=1)), []),
            (PatchTSModel(encoder_length=7, decoder_length=7, trainer_params=dict(max_epochs=1)), []),
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
            (NBeatsInterpretableModel(input_size=1, output_size=7, trainer_params=dict(max_epochs=1)), []),
            (NBeatsGenericModel(input_size=1, output_size=7, trainer_params=dict(max_epochs=1)), []),
        ],
    )
    def test_forecast_in_sample_full_no_target_failed_not_enough_context(self, model, transforms, example_tsds):
        with pytest.raises(ValueError, match="Given context isn't big enough"):
            self._test_forecast_in_sample_full_no_target(example_tsds, model, transforms)

    @pytest.mark.parametrize(
        "model, transforms",
        [
            (
                MLPModel(input_size=2, hidden_size=[10], decoder_length=7, trainer_params=dict(max_epochs=1)),
                [LagTransform(in_column="target", lags=[5, 6])],
            ),
        ],
    )
    def test_forecast_in_sample_full_no_target_failed_nans_nn(self, model, transforms, example_tsds):
        with pytest.raises(ValueError, match="There are NaNs in features"):
            self._test_forecast_in_sample_full_no_target(example_tsds, model, transforms)

    @to_be_fixed(raises=NotImplementedError, match="This model can't make forecast on history data")
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
        ],
    )
    def test_forecast_in_sample_full_no_target_failed_not_implemented_in_sample(self, model, transforms, example_tsds):
        self._test_forecast_in_sample_full_no_target(example_tsds, model, transforms)


class TestForecastInSampleFull:
    """Test forecast on full train dataset.

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
    def test_forecast_in_sample_full_failed_nans_sklearn(self, model, transforms, example_tsds):
        with pytest.raises(ValueError, match="Input contains NaN, infinity or a value too large"):
            _test_prediction_in_sample_full(example_tsds, model, transforms, method_name="forecast")

    @pytest.mark.parametrize(
        "model, transforms",
        [
            (
                MLPModel(input_size=2, hidden_size=[10], decoder_length=7, trainer_params=dict(max_epochs=1)),
                [LagTransform(in_column="target", lags=[2, 3])],
            ),
        ],
    )
    def test_forecast_in_sample_full_failed_nans_nn(self, model, transforms, example_tsds):
        with pytest.raises(ValueError, match="There are NaNs in features"):
            _test_prediction_in_sample_full(example_tsds, model, transforms, method_name="forecast")

    @pytest.mark.parametrize(
        "model, transforms",
        [
            (MovingAverageModel(window=3), []),
            (NaiveModel(lag=3), []),
            (SeasonalMovingAverageModel(), []),
            (DeadlineMovingAverageModel(window=1), []),
            (RNNModel(input_size=1, encoder_length=7, decoder_length=7, trainer_params=dict(max_epochs=1)), []),
            (PatchTSModel(encoder_length=7, decoder_length=7, trainer_params=dict(max_epochs=1)), []),
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
            (NBeatsInterpretableModel(input_size=1, output_size=7, trainer_params=dict(max_epochs=1)), []),
            (NBeatsGenericModel(input_size=1, output_size=7, trainer_params=dict(max_epochs=1)), []),
        ],
    )
    def test_forecast_in_sample_full_failed_not_enough_context(self, model, transforms, example_tsds):
        with pytest.raises(ValueError, match="Given context isn't big enough"):
            _test_prediction_in_sample_full(example_tsds, model, transforms, method_name="forecast")

    @to_be_fixed(raises=NotImplementedError, match="This model can't make forecast on history data")
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
        ],
    )
    def test_forecast_in_sample_full_not_implemented(self, model, transforms, example_tsds):
        _test_prediction_in_sample_full(example_tsds, model, transforms, method_name="forecast")


class TestForecastInSampleSuffixNoTarget:
    """Test forecast on suffix of train dataset where target is filled with NaNs.

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
        forecast_ts.transform(transforms)
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
            (NBeatsInterpretableModel(input_size=7, output_size=50, trainer_params=dict(max_epochs=1)), []),
            (NBeatsGenericModel(input_size=7, output_size=50, trainer_params=dict(max_epochs=1)), []),
        ],
    )
    def test_forecast_in_sample_suffix_no_target(self, model, transforms, example_tsds):
        self._test_forecast_in_sample_suffix_no_target(example_tsds, model, transforms, num_skip_points=50)

    @to_be_fixed(raises=NotImplementedError, match="This model can't make forecast on history data")
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
        ],
    )
    def test_forecast_in_sample_suffix_no_target_failed_not_implemented_in_sample(
        self, model, transforms, example_tsds
    ):
        self._test_forecast_in_sample_suffix_no_target(example_tsds, model, transforms, num_skip_points=50)


class TestForecastInSampleSuffix:
    """Test forecast on suffix of train dataset.

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
            (NBeatsInterpretableModel(input_size=7, output_size=50, trainer_params=dict(max_epochs=1)), []),
            (NBeatsGenericModel(input_size=7, output_size=50, trainer_params=dict(max_epochs=1)), []),
        ],
    )
    def test_forecast_in_sample_suffix(self, model, transforms, example_tsds):
        _test_prediction_in_sample_suffix(example_tsds, model, transforms, method_name="forecast", num_skip_points=50)

    @to_be_fixed(raises=NotImplementedError, match="This model can't make forecast on history data")
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
        forecast_full_ts = ts.make_future(
            future_steps=full_prediction_size, tail_steps=model.context_size, transforms=transforms
        )
        forecast_full_ts = make_forecast(model=model, ts=forecast_full_ts, prediction_size=full_prediction_size)

        # forecasting only prefix
        forecast_prefix_ts = ts.make_future(
            future_steps=full_prediction_size, tail_steps=model.context_size, transforms=transforms
        )
        forecast_prefix_ts.df = forecast_prefix_ts.df.iloc[:-prediction_size_diff]
        forecast_prefix_ts = make_forecast(model=model, ts=forecast_prefix_ts, prediction_size=prefix_prediction_size)

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
            (BATSModel(use_trend=True), []),
            (TBATSModel(use_trend=True), []),
            (StatsForecastARIMAModel(), []),
            (StatsForecastAutoARIMAModel(), []),
            (StatsForecastAutoCESModel(), []),
            (StatsForecastAutoETSModel(), []),
            (StatsForecastAutoThetaModel(), []),
            (RNNModel(input_size=1, encoder_length=7, decoder_length=7, trainer_params=dict(max_epochs=1)), []),
            (PatchTSModel(encoder_length=7, decoder_length=7, trainer_params=dict(max_epochs=1)), []),
            (
                MLPModel(input_size=2, hidden_size=[10], decoder_length=7, trainer_params=dict(max_epochs=1)),
                [LagTransform(in_column="target", lags=[5, 6])],
            ),
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
            (NBeatsInterpretableModel(input_size=7, output_size=7, trainer_params=dict(max_epochs=1)), []),
            (NBeatsGenericModel(input_size=7, output_size=7, trainer_params=dict(max_epochs=1)), []),
        ],
    )
    def test_forecast_out_sample_prefix(self, model, transforms, example_tsds):
        self._test_forecast_out_sample_prefix(example_tsds, model, transforms)

    @pytest.mark.parametrize(
        "model, transforms",
        [
            (
                DeepStateModel(
                    ssm=CompositeSSM(seasonal_ssms=[WeeklySeasonalitySSM()]),
                    input_size=1,
                    encoder_length=7,
                    decoder_length=7,
                    trainer_params=dict(max_epochs=1),
                ),
                [SegmentEncoderTransform()],
            )
        ],
    )
    def test_forecast_out_sample_prefix_failed_deep_state(self, model, transforms, example_tsds):
        """This test is expected to fail due to sampling procedure of DeepStateModel"""
        with pytest.raises(AssertionError):
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
        forecast_full_ts = ts.make_future(
            future_steps=full_prediction_size, tail_steps=model.context_size, transforms=transforms
        )
        forecast_full_ts = make_forecast(model=model, ts=forecast_full_ts, prediction_size=full_prediction_size)

        # forecasting only suffix
        forecast_gap_ts = ts.make_future(
            future_steps=full_prediction_size, tail_steps=model.context_size, transforms=transforms
        )
        if isinstance(model, get_args(ContextRequiredModelType)):
            # firstly we should forecast prefix to use it as a context
            forecast_prefix_ts = deepcopy(forecast_gap_ts)
            forecast_prefix_ts.df = forecast_prefix_ts.df.iloc[:-suffix_prediction_size]
            forecast_prefix_ts = model.forecast(forecast_prefix_ts, prediction_size=prediction_size_diff)
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
            (BATSModel(use_trend=True), []),
            (TBATSModel(use_trend=True), []),
            (MovingAverageModel(window=3), []),
            (SeasonalMovingAverageModel(), []),
            (NaiveModel(lag=3), []),
            (DeadlineMovingAverageModel(window=1), []),
            (PatchTSModel(encoder_length=7, decoder_length=7, trainer_params=dict(max_epochs=1)), []),
            (
                MLPModel(input_size=2, hidden_size=[10], decoder_length=7, trainer_params=dict(max_epochs=1)),
                [LagTransform(in_column="target", lags=[5, 6])],
            ),
        ],
    )
    def test_forecast_out_sample_suffix(self, model, transforms, example_tsds):
        self._test_forecast_out_sample_suffix(example_tsds, model, transforms)

    @pytest.mark.parametrize(
        "model, transforms",
        [
            (RNNModel(input_size=1, encoder_length=7, decoder_length=7, trainer_params=dict(max_epochs=1)), []),
        ],
    )
    def test_forecast_out_sample_suffix_failed_rnn(self, model, transforms, example_tsds):
        """This test is expected to fail due to autoregression in RNN.

        More about it in issue: https://github.com/tinkoff-ai/etna/issues/1087
        """
        with pytest.raises(AssertionError):
            self._test_forecast_out_sample_suffix(example_tsds, model, transforms)

    @pytest.mark.parametrize(
        "model, transforms",
        [
            (
                DeepStateModel(
                    ssm=CompositeSSM(seasonal_ssms=[WeeklySeasonalitySSM()]),
                    input_size=1,
                    encoder_length=7,
                    decoder_length=7,
                    trainer_params=dict(max_epochs=1),
                ),
                [SegmentEncoderTransform()],
            )
        ],
    )
    def test_forecast_out_sample_suffix_failed_deep_state(self, model, transforms, example_tsds):
        """This test is expected to fail due to sampling procedure of DeepStateModel"""
        with pytest.raises(AssertionError):
            self._test_forecast_out_sample_suffix(example_tsds, model, transforms)

    @pytest.mark.parametrize(
        "model,transforms",
        (
            (NBeatsInterpretableModel(input_size=7, output_size=7, trainer_params=dict(max_epochs=1)), []),
            (NBeatsGenericModel(input_size=7, output_size=7, trainer_params=dict(max_epochs=1)), []),
        ),
    )
    def test_forecast_out_sample_suffix_failed_nbeats(self, model, transforms, example_tsds):
        """This test is expected to fail due to windowed view on data in N-BEATS"""
        with pytest.raises(AssertionError):
            self._test_forecast_out_sample_suffix(example_tsds, model, transforms)

    @to_be_fixed(
        raises=NotImplementedError,
        match="This model can't make forecast on out-of-sample data that goes after training data with a gap",
    )
    @pytest.mark.parametrize(
        "model, transforms",
        [
            (StatsForecastARIMAModel(), []),
            (StatsForecastAutoARIMAModel(), []),
            (StatsForecastAutoCESModel(), []),
            (StatsForecastAutoETSModel(), []),
            (StatsForecastAutoThetaModel(), []),
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
        ],
    )
    def test_forecast_out_sample_suffix_failed_not_implemented(self, model, transforms, example_tsds):
        self._test_forecast_out_sample_suffix(example_tsds, model, transforms)


class TestForecastMixedInOutSample:
    """Test forecast on mixture of in-sample and out-sample.

    Expected that there are no NaNs after prediction and targets are changed compared to original.
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
        forecast_full_ts.transform(transforms)
        forecast_full_ts.df = forecast_full_ts.df.iloc[(num_skip_points - model.context_size) :]
        full_prediction_size = len(forecast_full_ts.index) - model.context_size
        forecast_full_ts = make_forecast(model=model, ts=forecast_full_ts, prediction_size=full_prediction_size)

        # checking
        forecast_full_df = forecast_full_ts.to_pandas(flatten=True)
        assert not np.any(forecast_full_df["target"].isna())
        original_target = TSDataset.to_flatten(df_full.iloc[(num_skip_points - model.context_size) :])["target"]
        assert not forecast_full_df["target"].equals(original_target)

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
            (NBeatsInterpretableModel(input_size=7, output_size=55, trainer_params=dict(max_epochs=1)), []),
            (NBeatsGenericModel(input_size=7, output_size=55, trainer_params=dict(max_epochs=1)), []),
        ],
    )
    def test_forecast_mixed_in_out_sample(self, model, transforms, example_tsds):
        self._test_forecast_mixed_in_out_sample(example_tsds, model, transforms)

    @to_be_fixed(raises=NotImplementedError, match="This model can't make forecast on history data")
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
        ],
    )
    def test_forecast_mixed_in_out_sample_failed_not_implemented_in_sample(self, model, transforms, example_tsds):
        self._test_forecast_mixed_in_out_sample(example_tsds, model, transforms)


class TestForecastSubsetSegments:
    """Test forecast on subset of segments.

    Expected that predictions on subset of segments match subset of predictions on full dataset.
    """

    def _test_forecast_subset_segments(self, ts, model, transforms, segments, prediction_size=5):
        # select subset of tsdataset
        segments = list(set(segments))
        subset_ts = select_segments_subset(ts=deepcopy(ts), segments=segments)

        # fitting
        ts.fit_transform(transforms)
        model.fit(ts)

        # forecasting full
        forecast_full_ts = ts.make_future(
            future_steps=prediction_size, tail_steps=model.context_size, transforms=transforms
        )
        forecast_full_ts = make_forecast(model=model, ts=forecast_full_ts, prediction_size=prediction_size)

        # forecasting subset of segments
        forecast_subset_ts = subset_ts.make_future(
            future_steps=prediction_size, tail_steps=model.context_size, transforms=transforms
        )
        forecast_subset_ts = make_forecast(model=model, ts=forecast_subset_ts, prediction_size=prediction_size)

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
            (NBeatsInterpretableModel(input_size=7, output_size=7, trainer_params=dict(max_epochs=1)), []),
            (NBeatsGenericModel(input_size=7, output_size=7, trainer_params=dict(max_epochs=1)), []),
        ],
    )
    def test_forecast_subset_segments(self, model, transforms, example_tsds):
        self._test_forecast_subset_segments(example_tsds, model, transforms, segments=["segment_2"])

    @pytest.mark.parametrize(
        "model, transforms",
        [
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
        ],
    )
    def test_forecast_subset_segments_failed_deep_state(self, model, transforms, example_tsds):
        with pytest.raises(AssertionError):
            self._test_forecast_subset_segments(example_tsds, model, transforms, segments=["segment_2"])

    @to_be_fixed(raises=AssertionError)
    # issue with explanation: https://github.com/tinkoff-ai/etna/issues/1089
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
        ],
    )
    def test_forecast_subset_segments_failed_assertion_error(self, model, transforms, example_tsds):
        self._test_forecast_subset_segments(example_tsds, model, transforms, segments=["segment_2"])


class TestForecastNewSegments:
    """Test forecast on new segments.

    Expected that target values are filled after prediction.
    """

    def _test_forecast_new_segments(self, ts, model, transforms, train_segments, prediction_size=5):
        # create tsdataset with new segments
        train_segments = list(set(train_segments))
        forecast_segments = list(set(ts.segments) - set(train_segments))
        train_ts = select_segments_subset(ts=deepcopy(ts), segments=train_segments)
        test_ts = select_segments_subset(ts=deepcopy(ts), segments=forecast_segments)

        # fitting
        train_ts.fit_transform(transforms)
        model.fit(train_ts)

        # forecasting
        forecast_ts = test_ts.make_future(
            future_steps=prediction_size, tail_steps=model.context_size, transforms=transforms
        )
        forecast_ts = make_forecast(model=model, ts=forecast_ts, prediction_size=prediction_size)

        # checking
        forecast_df = forecast_ts.to_pandas(flatten=True)
        assert not np.any(forecast_df["target"].isna())

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
            (RNNModel(input_size=1, encoder_length=7, decoder_length=7, trainer_params=dict(max_epochs=1)), []),
            (PatchTSModel(encoder_length=7, decoder_length=7, trainer_params=dict(max_epochs=1)), []),
            (
                MLPModel(input_size=2, hidden_size=[10], decoder_length=7, trainer_params=dict(max_epochs=1)),
                [LagTransform(in_column="target", lags=[5, 6])],
            ),
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
    def test_forecast_new_segments(self, model, transforms, example_tsds):
        self._test_forecast_new_segments(example_tsds, model, transforms, train_segments=["segment_1"])

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
    def test_forecast_new_segments_failed_per_segment(self, model, transforms, example_tsds):
        with pytest.raises(NotImplementedError, match="Per-segment models can't make predictions on new segments"):
            self._test_forecast_new_segments(example_tsds, model, transforms, train_segments=["segment_1"])
