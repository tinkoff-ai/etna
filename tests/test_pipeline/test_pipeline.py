from copy import deepcopy
from typing import Dict
from typing import List
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from etna.datasets import TSDataset
from etna.datasets import generate_ar_df
from etna.distributions import CategoricalDistribution
from etna.distributions import FloatDistribution
from etna.distributions import IntDistribution
from etna.metrics import MAE
from etna.metrics import MSE
from etna.metrics import SMAPE
from etna.metrics import MetricAggregationMode
from etna.metrics import Width
from etna.models import CatBoostMultiSegmentModel
from etna.models import LinearPerSegmentModel
from etna.models import MovingAverageModel
from etna.models import NaiveModel
from etna.models import ProphetModel
from etna.models import SARIMAXModel
from etna.models import SeasonalMovingAverageModel
from etna.models.base import NonPredictionIntervalContextIgnorantAbstractModel
from etna.models.base import NonPredictionIntervalContextRequiredAbstractModel
from etna.models.base import PredictionIntervalContextIgnorantAbstractModel
from etna.models.base import PredictionIntervalContextRequiredAbstractModel
from etna.pipeline import FoldMask
from etna.pipeline import Pipeline
from etna.pipeline.base import CrossValidationMode
from etna.transforms import AddConstTransform
from etna.transforms import DateFlagsTransform
from etna.transforms import DifferencingTransform
from etna.transforms import FilterFeaturesTransform
from etna.transforms import LagTransform
from etna.transforms import LogTransform
from etna.transforms import TimeSeriesImputerTransform
from tests.test_pipeline.utils import assert_pipeline_equals_loaded_original
from tests.test_pipeline.utils import assert_pipeline_forecasts_given_ts
from tests.test_pipeline.utils import assert_pipeline_forecasts_given_ts_with_prediction_intervals
from tests.utils import DummyMetric

DEFAULT_METRICS = [MAE(mode=MetricAggregationMode.per_segment)]


@pytest.fixture
def ts_with_feature():
    periods = 100
    df = generate_ar_df(
        start_time="2019-01-01", periods=periods, ar_coef=[1], sigma=1, n_segments=2, random_seed=0, freq="D"
    )
    df_feature = generate_ar_df(
        start_time="2019-01-01", periods=periods, ar_coef=[0.9], sigma=2, n_segments=2, random_seed=42, freq="D"
    )
    df["feature_1"] = df_feature["target"].apply(lambda x: abs(x))
    df = TSDataset.to_dataset(df)
    ts = TSDataset(df, freq="D")
    return ts


@pytest.mark.parametrize("horizon", ([1]))
def test_init_pass(horizon):
    """Check that Pipeline initialization works correctly in case of valid parameters."""
    pipeline = Pipeline(model=LinearPerSegmentModel(), transforms=[], horizon=horizon)
    assert pipeline.horizon == horizon


@pytest.mark.parametrize("horizon", ([-1]))
def test_init_fail(horizon):
    """Check that Pipeline initialization works correctly in case of invalid parameters."""
    with pytest.raises(ValueError, match="At least one point in the future is expected"):
        _ = Pipeline(
            model=LinearPerSegmentModel(),
            transforms=[],
            horizon=horizon,
        )


def test_fit(example_tsds):
    """Test that Pipeline correctly transforms dataset on fit stage."""
    original_ts = deepcopy(example_tsds)
    model = LinearPerSegmentModel()
    transforms = [AddConstTransform(in_column="target", value=10, inplace=True), DateFlagsTransform()]
    pipeline = Pipeline(model=model, transforms=transforms, horizon=5)
    pipeline.fit(example_tsds)
    original_ts.fit_transform(transforms)
    original_ts.inverse_transform(transforms)
    assert np.all(original_ts.df.values == pipeline.ts.df.values)


@patch("etna.pipeline.pipeline.Pipeline._forecast")
def test_forecast_without_intervals_calls_private_forecast(private_forecast, example_tsds):
    model = LinearPerSegmentModel()
    transforms = [AddConstTransform(in_column="target", value=10, inplace=True), DateFlagsTransform()]
    pipeline = Pipeline(model=model, transforms=transforms, horizon=5)
    pipeline.fit(example_tsds)
    _ = pipeline.forecast()

    private_forecast.assert_called()


@pytest.mark.parametrize(
    "model_fixture",
    (
        "non_prediction_interval_context_ignorant_dummy_model",
        "non_prediction_interval_context_required_dummy_model",
        "prediction_interval_context_ignorant_dummy_model",
        "prediction_interval_context_required_dummy_model",
    ),
)
def test_forecast_return_components(
    example_tsds, model_fixture, request, expected_component_a=10, expected_component_b=90
):
    model = request.getfixturevalue(model_fixture)
    pipeline = Pipeline(model=model)
    pipeline.fit(example_tsds)
    forecast = pipeline.forecast(return_components=True)
    assert sorted(forecast.target_components_names) == sorted(["target_component_a", "target_component_b"])

    target_components_df = TSDataset.to_flatten(forecast.get_target_components())
    assert (target_components_df["target_component_a"] == expected_component_a).all()
    assert (target_components_df["target_component_b"] == expected_component_b).all()


@pytest.mark.parametrize(
    "model_class", [NonPredictionIntervalContextIgnorantAbstractModel, PredictionIntervalContextIgnorantAbstractModel]
)
def test_private_forecast_context_ignorant_model(model_class):
    ts = MagicMock(spec=TSDataset)
    model = MagicMock(spec=model_class)

    pipeline = Pipeline(model=model, horizon=5)
    pipeline.fit(ts)
    _ = pipeline._forecast(ts=ts, return_components=False)

    ts.make_future.assert_called_with(future_steps=pipeline.horizon, transforms=())
    model.forecast.assert_called_with(ts=ts.make_future(), return_components=False)


@pytest.mark.parametrize(
    "model_class", [NonPredictionIntervalContextRequiredAbstractModel, PredictionIntervalContextRequiredAbstractModel]
)
def test_private_forecast_context_required_model(model_class):
    ts = MagicMock(spec=TSDataset)
    model = MagicMock(spec=model_class)

    pipeline = Pipeline(model=model, horizon=5)
    pipeline.fit(ts)
    _ = pipeline._forecast(ts=ts, return_components=False)

    ts.make_future.assert_called_with(future_steps=pipeline.horizon, transforms=(), tail_steps=model.context_size)
    model.forecast.assert_called_with(ts=ts.make_future(), prediction_size=pipeline.horizon, return_components=False)


def test_forecast_with_intervals_prediction_interval_context_ignorant_model():
    ts = MagicMock(spec=TSDataset)
    model = MagicMock(spec=PredictionIntervalContextIgnorantAbstractModel)

    pipeline = Pipeline(model=model, horizon=5)
    pipeline.fit(ts)
    _ = pipeline.forecast(prediction_interval=True, quantiles=(0.025, 0.975))

    ts.make_future.assert_called_with(future_steps=pipeline.horizon, transforms=())
    model.forecast.assert_called_with(
        ts=ts.make_future(), prediction_interval=True, quantiles=(0.025, 0.975), return_components=False
    )


def test_forecast_with_intervals_prediction_interval_context_required_model():
    ts = MagicMock(spec=TSDataset)
    model = MagicMock(spec=PredictionIntervalContextRequiredAbstractModel)

    pipeline = Pipeline(model=model, horizon=5)
    pipeline.fit(ts)
    _ = pipeline.forecast(prediction_interval=True, quantiles=(0.025, 0.975))

    ts.make_future.assert_called_with(future_steps=pipeline.horizon, transforms=(), tail_steps=model.context_size)
    model.forecast.assert_called_with(
        ts=ts.make_future(),
        prediction_size=pipeline.horizon,
        prediction_interval=True,
        quantiles=(0.025, 0.975),
        return_components=False,
    )


@patch("etna.pipeline.base.BasePipeline.forecast")
@pytest.mark.parametrize(
    "model_class",
    [NonPredictionIntervalContextIgnorantAbstractModel, NonPredictionIntervalContextRequiredAbstractModel],
)
def test_forecast_with_intervals_other_model(base_forecast, model_class):
    ts = MagicMock(spec=TSDataset)
    model = MagicMock(spec=model_class)

    pipeline = Pipeline(model=model, horizon=5)
    pipeline.fit(ts)
    _ = pipeline.forecast(prediction_interval=True, quantiles=(0.025, 0.975))
    base_forecast.assert_called_with(
        ts=ts, prediction_interval=True, quantiles=(0.025, 0.975), n_folds=3, return_components=False
    )


def test_forecast_values(example_tsds):
    """Test that the forecast from the Pipeline generates correct values."""
    original_ts = deepcopy(example_tsds)

    model = LinearPerSegmentModel()
    transforms = [AddConstTransform(in_column="target", value=10, inplace=True), DateFlagsTransform()]
    pipeline = Pipeline(model=model, transforms=transforms, horizon=5)
    pipeline.fit(example_tsds)
    forecast_pipeline = pipeline.forecast()

    original_ts.fit_transform(transforms)
    model.fit(original_ts)
    future = original_ts.make_future(5, transforms=transforms)
    forecast_manual = model.forecast(future)
    forecast_manual.inverse_transform(transforms)

    pd.testing.assert_frame_equal(forecast_pipeline.to_pandas(), forecast_manual.to_pandas())


@pytest.mark.parametrize(
    "quantiles,prediction_interval_cv,error_msg",
    (
        [
            ([0.05, 1.5], 2, "Quantile should be a number from"),
            ([0.025, 0.975], 0, "Folds number should be a positive number, 0 given"),
        ]
    ),
)
def test_forecast_prediction_interval_incorrect_parameters(
    example_tsds, catboost_pipeline, quantiles, prediction_interval_cv, error_msg
):
    catboost_pipeline.fit(ts=deepcopy(example_tsds))
    with pytest.raises(ValueError, match=error_msg):
        _ = catboost_pipeline.forecast(quantiles=quantiles, n_folds=prediction_interval_cv)


@pytest.mark.parametrize("model", (ProphetModel(), SARIMAXModel()))
def test_forecast_prediction_interval_builtin(example_tsds, model):
    """Test that forecast method uses built-in prediction intervals for the listed models."""
    np.random.seed(1234)
    pipeline = Pipeline(model=model, transforms=[], horizon=5)
    pipeline.fit(example_tsds)
    forecast_pipeline = pipeline.forecast(prediction_interval=True)

    np.random.seed(1234)
    model = model.fit(example_tsds)
    future = example_tsds.make_future(5)
    forecast_model = model.forecast(ts=future, prediction_interval=True, return_components=False)

    assert forecast_model.df.equals(forecast_pipeline.df)


@pytest.mark.parametrize("model", (MovingAverageModel(), LinearPerSegmentModel()))
def test_forecast_prediction_interval_not_builtin(example_tsds, model):
    """Test the forecast interface for the models without built-in prediction intervals."""
    pipeline = Pipeline(model=model, transforms=[DateFlagsTransform()], horizon=5)
    pipeline.fit(example_tsds)
    forecast = pipeline.forecast(prediction_interval=True, quantiles=[0.025, 0.975])
    for segment in forecast.segments:
        segment_slice = forecast[:, segment, :][segment]
        assert {"target_0.025", "target_0.975", "target"}.issubset(segment_slice.columns)
        assert (segment_slice["target_0.975"] - segment_slice["target_0.025"] >= 0).all()


@pytest.mark.parametrize("model", (MovingAverageModel(), LinearPerSegmentModel()))
def test_forecast_prediction_interval_not_builtin_with_nans_warning(example_tsds, model):
    example_tsds.df.loc[example_tsds.index[-2], pd.IndexSlice["segment_1", "target"]] = None

    pipeline = Pipeline(model=model, transforms=[DateFlagsTransform()], horizon=5)
    pipeline.fit(example_tsds)
    with pytest.warns(UserWarning, match="There are NaNs in target on time span from .* to .*"):
        _ = pipeline.forecast(prediction_interval=True, quantiles=[0.025, 0.975])


@pytest.mark.parametrize("model", (MovingAverageModel(), LinearPerSegmentModel()))
def test_forecast_prediction_interval_not_builtin_with_nans_error(example_tsds, model):
    example_tsds.df.loc[example_tsds.index[-20:-1], pd.IndexSlice["segment_1", "target"]] = None

    pipeline = Pipeline(model=model, transforms=[DateFlagsTransform()], horizon=5)
    pipeline.fit(example_tsds)
    with pytest.raises(
        ValueError, match="There aren't enough target values to evaluate prediction intervals on history"
    ):
        _ = pipeline.forecast(prediction_interval=True, quantiles=[0.025, 0.975])


def test_forecast_prediction_interval_correct_values(splited_piecewise_constant_ts):
    """Test that the prediction interval for piecewise-constant dataset is correct."""
    train, test = splited_piecewise_constant_ts
    pipeline = Pipeline(model=NaiveModel(lag=1), transforms=[], horizon=5)
    pipeline.fit(train)
    forecast = pipeline.forecast(prediction_interval=True)
    assert np.allclose(forecast.df.values, test.df.values)


@pytest.mark.parametrize("quantiles_narrow,quantiles_wide", ([([0.2, 0.8], [0.025, 0.975])]))
def test_forecast_prediction_interval_size(example_tsds, quantiles_narrow, quantiles_wide):
    """Test that narrow quantile levels gives more narrow interval than wide quantile levels."""
    pipeline = Pipeline(model=MovingAverageModel(), transforms=[], horizon=5)
    pipeline.fit(example_tsds)
    forecast = pipeline.forecast(prediction_interval=True, quantiles=quantiles_narrow)
    narrow_interval_length = (
        forecast[:, :, f"target_{quantiles_narrow[1]}"].values - forecast[:, :, f"target_{quantiles_narrow[0]}"].values
    )

    pipeline = Pipeline(model=MovingAverageModel(), transforms=[], horizon=5)
    pipeline.fit(example_tsds)
    forecast = pipeline.forecast(prediction_interval=True, quantiles=quantiles_wide)
    wide_interval_length = (
        forecast[:, :, f"target_{quantiles_wide[1]}"].values - forecast[:, :, f"target_{quantiles_wide[0]}"].values
    )

    assert (narrow_interval_length <= wide_interval_length).all()


def test_forecast_prediction_interval_noise(constant_ts, constant_noisy_ts):
    """Test that prediction interval for noisy dataset is wider then for the dataset without noise."""
    pipeline = Pipeline(model=MovingAverageModel(), transforms=[], horizon=5)
    pipeline.fit(constant_ts)
    forecast = pipeline.forecast(prediction_interval=True, quantiles=[0.025, 0.975])
    constant_interval_length = forecast[:, :, "target_0.975"].values - forecast[:, :, "target_0.025"].values

    pipeline = Pipeline(model=MovingAverageModel(), transforms=[], horizon=5)
    pipeline.fit(constant_noisy_ts)
    forecast = pipeline.forecast(prediction_interval=True)
    noisy_interval_length = forecast[:, :, "target_0.975"].values - forecast[:, :, "target_0.025"].values

    assert (constant_interval_length <= noisy_interval_length).all()


@pytest.mark.parametrize("n_folds", (0, -1))
def test_invalid_n_folds(catboost_pipeline: Pipeline, n_folds: int, example_tsdf: TSDataset):
    """Test Pipeline.backtest behavior in case of invalid n_folds."""
    with pytest.raises(ValueError, match="Folds number should be a positive number"):
        _ = catboost_pipeline.backtest(ts=example_tsdf, metrics=DEFAULT_METRICS, n_folds=n_folds)


@pytest.mark.parametrize(
    "min_size, n_folds, horizon, stride",
    [
        (1, 10, 1, 1),
        (9, 10, 1, 1),
        (10, 10, 2, 1),
        (19, 10, 2, 2),
        (28, 10, 2, 3),
    ],
)
def test_invalid_backtest_dataset_size(min_size, n_folds, horizon, stride):
    """Test Pipeline.backtest behavior in case of too small dataframe for given number of folds."""
    df = generate_ar_df(start_time="2020-01-01", periods=100, n_segments=2, freq="D")
    df_wide = TSDataset.to_dataset(df)
    to_remove = len(df_wide) - min_size
    df_wide.iloc[:to_remove, 0] = np.NaN
    ts = TSDataset(df=df_wide, freq="D")
    pipeline = Pipeline(model=NaiveModel(lag=horizon), horizon=horizon)

    with pytest.raises(ValueError, match="All the series from feature dataframe should contain at least .* timestamps"):
        _ = pipeline.backtest(ts=ts, n_folds=n_folds, stride=stride, metrics=DEFAULT_METRICS)


def test_invalid_backtest_metrics_empty(catboost_pipeline: Pipeline, example_tsdf: TSDataset):
    """Test Pipeline.backtest behavior in case of empty metrics."""
    with pytest.raises(ValueError, match="At least one metric required"):
        _ = catboost_pipeline.backtest(ts=example_tsdf, metrics=[], n_folds=2)


def test_invalid_backtest_metrics_macro(catboost_pipeline: Pipeline, example_tsdf: TSDataset):
    """Test Pipeline.backtest behavior in case of macro metrics."""
    with pytest.raises(ValueError, match="All the metrics should be in"):
        _ = catboost_pipeline.backtest(ts=example_tsdf, metrics=[MAE(mode=MetricAggregationMode.macro)], n_folds=2)


def test_invalid_backtest_mode_set_on_fold_mask(catboost_pipeline: Pipeline, example_tsdf: TSDataset):
    """Test Pipeline.backtest behavior on setting mode with fold masks."""
    masks = [
        FoldMask(
            first_train_timestamp="2020-01-01",
            last_train_timestamp="2020-04-03",
            target_timestamps=["2020-04-04", "2020-04-05", "2020-04-06"],
        ),
        FoldMask(
            first_train_timestamp="2020-01-01",
            last_train_timestamp="2020-04-06",
            target_timestamps=["2020-04-07", "2020-04-08", "2020-04-09"],
        ),
    ]
    with pytest.raises(ValueError, match="Mode shouldn't be set if n_folds are fold masks"):
        _ = catboost_pipeline.backtest(ts=example_tsdf, n_folds=masks, mode="expand", metrics=DEFAULT_METRICS)


def test_invalid_backtest_stride_set_on_fold_mask(catboost_pipeline: Pipeline, example_tsdf: TSDataset):
    """Test Pipeline.backtest behavior on setting stride with fold masks."""
    masks = [
        FoldMask(
            first_train_timestamp="2020-01-01",
            last_train_timestamp="2020-04-03",
            target_timestamps=["2020-04-04", "2020-04-05", "2020-04-06"],
        ),
        FoldMask(
            first_train_timestamp="2020-01-01",
            last_train_timestamp="2020-04-06",
            target_timestamps=["2020-04-07", "2020-04-08", "2020-04-09"],
        ),
    ]
    with pytest.raises(ValueError, match="Stride shouldn't be set if n_folds are fold masks"):
        _ = catboost_pipeline.backtest(ts=example_tsdf, n_folds=masks, stride=2, metrics=DEFAULT_METRICS)


@pytest.mark.parametrize("stride", [-1, 0])
def test_invalid_backtest_stride_not_positive(stride, catboost_pipeline: Pipeline, example_tsdf: TSDataset):
    """Test Pipeline.backtest behavior on setting not positive stride."""
    with pytest.raises(ValueError, match="Stride should be a positive number, .* given"):
        _ = catboost_pipeline.backtest(ts=example_tsdf, n_folds=3, stride=stride, metrics=DEFAULT_METRICS)


@pytest.mark.parametrize(
    "aggregate_metrics,expected_columns",
    (
        (
            False,
            ["fold_number", "MAE", "MSE", "segment", "SMAPE", DummyMetric("per-segment", alpha=0.0).__repr__()],
        ),
        (
            True,
            ["MAE", "MSE", "segment", "SMAPE", DummyMetric("per-segment", alpha=0.0).__repr__()],
        ),
    ),
)
def test_backtest_metrics_interface(
    catboost_pipeline: Pipeline, aggregate_metrics: bool, expected_columns: List[str], big_daily_example_tsdf: TSDataset
):
    """Check that Pipeline.backtest returns metrics in correct format."""
    metrics_df, _, _ = catboost_pipeline.backtest(
        ts=big_daily_example_tsdf,
        aggregate_metrics=aggregate_metrics,
        metrics=[MAE("per-segment"), MSE("per-segment"), SMAPE("per-segment"), DummyMetric("per-segment", alpha=0.0)],
    )
    assert sorted(expected_columns) == sorted(metrics_df.columns)


@pytest.mark.parametrize(
    "ts_fixture",
    [
        "big_daily_example_tsdf",
        "example_tsdf",
    ],
)
def test_backtest_forecasts_columns(ts_fixture, catboost_pipeline, request):
    """Check that Pipeline.backtest returns forecasts in correct format."""
    ts = request.getfixturevalue(ts_fixture)
    _, forecast, _ = catboost_pipeline.backtest(ts=ts, metrics=DEFAULT_METRICS)
    expected_columns = sorted(
        ["regressor_lag_feature_10", "regressor_lag_feature_11", "regressor_lag_feature_12", "fold_number", "target"]
    )
    assert expected_columns == sorted(set(forecast.columns.get_level_values("feature")))


@pytest.mark.parametrize(
    "n_folds, horizon, expected_timestamps",
    [
        (2, 3, [-6, -5, -4, -3, -2, -1]),
        (2, 5, [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1]),
        (
            [
                FoldMask(
                    first_train_timestamp=pd.Timestamp("2020-01-01"),
                    last_train_timestamp=pd.Timestamp("2020-01-31 14:00"),
                    target_timestamps=[pd.Timestamp("2020-01-31 17:00")],
                ),
                FoldMask(
                    first_train_timestamp=pd.Timestamp("2020-01-01"),
                    last_train_timestamp=pd.Timestamp("2020-01-31 19:00"),
                    target_timestamps=[pd.Timestamp("2020-01-31 22:00")],
                ),
            ],
            5,
            [-8, -3],
        ),
    ],
)
def test_backtest_forecasts_timestamps(n_folds, horizon, expected_timestamps, example_tsdf):
    """Check that Pipeline.backtest returns forecasts with expected timestamps."""
    pipeline = Pipeline(model=NaiveModel(lag=horizon), horizon=horizon)
    _, forecast, _ = pipeline.backtest(ts=example_tsdf, metrics=DEFAULT_METRICS, n_folds=n_folds)
    timestamp = example_tsdf.index

    np.testing.assert_array_equal(forecast.index, timestamp[expected_timestamps])


@pytest.mark.parametrize(
    "n_folds, horizon, stride, expected_timestamps",
    [
        (2, 3, 3, [-6, -5, -4, -3, -2, -1]),
        (2, 3, 1, [-4, -3, -2, -3, -2, -1]),
        (2, 3, 5, [-8, -7, -6, -3, -2, -1]),
    ],
)
def test_backtest_forecasts_timestamps_with_stride(n_folds, horizon, stride, expected_timestamps, example_tsdf):
    """Check that Pipeline.backtest with stride returns forecasts with expected timestamps."""
    pipeline = Pipeline(model=NaiveModel(lag=horizon), horizon=horizon)
    _, forecast, _ = pipeline.backtest(ts=example_tsdf, metrics=DEFAULT_METRICS, n_folds=n_folds, stride=stride)
    timestamp = example_tsdf.index

    np.testing.assert_array_equal(forecast.index, timestamp[expected_timestamps])


@pytest.mark.parametrize(
    "ts_fixture, n_folds",
    [
        ("big_daily_example_tsdf", 1),
        ("big_daily_example_tsdf", 2),
        ("example_tsdf", 1),
        ("example_tsdf", 2),
    ],
)
def test_backtest_fold_info_format(ts_fixture, n_folds, request):
    """Check that Pipeline.backtest returns info dataframe in correct format."""
    ts = request.getfixturevalue(ts_fixture)
    pipeline = Pipeline(model=NaiveModel(lag=7), horizon=7)
    _, _, info_df = pipeline.backtest(ts=ts, metrics=DEFAULT_METRICS, n_folds=n_folds)

    expected_folds = pd.Series(np.arange(n_folds))
    pd.testing.assert_series_equal(info_df["fold_number"], expected_folds, check_names=False)
    expected_columns = ["fold_number", "test_end_time", "test_start_time", "train_end_time", "train_start_time"]
    assert expected_columns == sorted(info_df.columns)


@pytest.mark.parametrize(
    "mode, n_folds, refit, horizon, stride, expected_train_starts, expected_train_ends, expected_test_starts, expected_test_ends",
    [
        ("expand", 3, True, 7, None, [0, 0, 0], [-22, -15, -8], [-21, -14, -7], [-15, -8, -1]),
        ("expand", 3, True, 7, 1, [0, 0, 0], [-10, -9, -8], [-9, -8, -7], [-3, -2, -1]),
        ("expand", 3, True, 7, 10, [0, 0, 0], [-28, -18, -8], [-27, -17, -7], [-21, -11, -1]),
        ("expand", 3, False, 7, None, [0, 0, 0], [-22, -22, -22], [-21, -14, -7], [-15, -8, -1]),
        ("expand", 3, False, 7, 1, [0, 0, 0], [-10, -10, -10], [-9, -8, -7], [-3, -2, -1]),
        ("expand", 3, False, 7, 10, [0, 0, 0], [-28, -28, -28], [-27, -17, -7], [-21, -11, -1]),
        ("expand", 1, 1, 7, None, [0], [-8], [-7], [-1]),
        ("expand", 1, 2, 7, None, [0], [-8], [-7], [-1]),
        ("expand", 3, 1, 7, None, [0, 0, 0], [-22, -15, -8], [-21, -14, -7], [-15, -8, -1]),
        ("expand", 3, 2, 7, None, [0, 0, 0], [-22, -22, -8], [-21, -14, -7], [-15, -8, -1]),
        ("expand", 3, 3, 7, None, [0, 0, 0], [-22, -22, -22], [-21, -14, -7], [-15, -8, -1]),
        ("expand", 3, 4, 7, None, [0, 0, 0], [-22, -22, -22], [-21, -14, -7], [-15, -8, -1]),
        ("expand", 4, 1, 7, None, [0, 0, 0, 0], [-29, -22, -15, -8], [-28, -21, -14, -7], [-22, -15, -8, -1]),
        ("expand", 4, 2, 7, None, [0, 0, 0, 0], [-29, -29, -15, -15], [-28, -21, -14, -7], [-22, -15, -8, -1]),
        ("expand", 4, 2, 7, 1, [0, 0, 0, 0], [-11, -11, -9, -9], [-10, -9, -8, -7], [-4, -3, -2, -1]),
        ("expand", 4, 2, 7, 10, [0, 0, 0, 0], [-38, -38, -18, -18], [-37, -27, -17, -7], [-31, -21, -11, -1]),
        ("expand", 4, 3, 7, None, [0, 0, 0, 0], [-29, -29, -29, -8], [-28, -21, -14, -7], [-22, -15, -8, -1]),
        ("expand", 4, 4, 7, None, [0, 0, 0, 0], [-29, -29, -29, -29], [-28, -21, -14, -7], [-22, -15, -8, -1]),
        ("expand", 4, 5, 7, None, [0, 0, 0, 0], [-29, -29, -29, -29], [-28, -21, -14, -7], [-22, -15, -8, -1]),
        ("constant", 3, True, 7, None, [0, 7, 14], [-22, -15, -8], [-21, -14, -7], [-15, -8, -1]),
        ("constant", 3, True, 7, 1, [0, 1, 2], [-10, -9, -8], [-9, -8, -7], [-3, -2, -1]),
        ("constant", 3, True, 7, 10, [0, 10, 20], [-28, -18, -8], [-27, -17, -7], [-21, -11, -1]),
        ("constant", 3, False, 7, None, [0, 0, 0], [-22, -22, -22], [-21, -14, -7], [-15, -8, -1]),
        ("constant", 3, False, 7, 1, [0, 0, 0], [-10, -10, -10], [-9, -8, -7], [-3, -2, -1]),
        ("constant", 3, False, 7, 10, [0, 0, 0], [-28, -28, -28], [-27, -17, -7], [-21, -11, -1]),
        ("constant", 1, 1, 7, None, [0], [-8], [-7], [-1]),
        ("constant", 1, 2, 7, None, [0], [-8], [-7], [-1]),
        ("constant", 3, 1, 7, None, [0, 7, 14], [-22, -15, -8], [-21, -14, -7], [-15, -8, -1]),
        ("constant", 3, 2, 7, None, [0, 0, 14], [-22, -22, -8], [-21, -14, -7], [-15, -8, -1]),
        ("constant", 3, 3, 7, None, [0, 0, 0], [-22, -22, -22], [-21, -14, -7], [-15, -8, -1]),
        ("constant", 3, 4, 7, None, [0, 0, 0], [-22, -22, -22], [-21, -14, -7], [-15, -8, -1]),
        ("constant", 4, 1, 7, None, [0, 7, 14, 21], [-29, -22, -15, -8], [-28, -21, -14, -7], [-22, -15, -8, -1]),
        ("constant", 4, 2, 7, None, [0, 0, 14, 14], [-29, -29, -15, -15], [-28, -21, -14, -7], [-22, -15, -8, -1]),
        ("constant", 4, 2, 7, 1, [0, 0, 2, 2], [-11, -11, -9, -9], [-10, -9, -8, -7], [-4, -3, -2, -1]),
        ("constant", 4, 2, 7, 10, [0, 0, 20, 20], [-38, -38, -18, -18], [-37, -27, -17, -7], [-31, -21, -11, -1]),
        ("constant", 4, 3, 7, None, [0, 0, 0, 21], [-29, -29, -29, -8], [-28, -21, -14, -7], [-22, -15, -8, -1]),
        ("constant", 4, 4, 7, None, [0, 0, 0, 0], [-29, -29, -29, -29], [-28, -21, -14, -7], [-22, -15, -8, -1]),
        ("constant", 4, 5, 7, None, [0, 0, 0, 0], [-29, -29, -29, -29], [-28, -21, -14, -7], [-22, -15, -8, -1]),
        (
            None,
            [
                FoldMask(
                    first_train_timestamp=None,
                    last_train_timestamp=pd.Timestamp("2020-01-31 10:00"),
                    target_timestamps=[pd.Timestamp("2020-01-31 14:00")],
                ),
                FoldMask(
                    first_train_timestamp=None,
                    last_train_timestamp=pd.Timestamp("2020-01-31 17:00"),
                    target_timestamps=[pd.Timestamp("2020-01-31 21:00")],
                ),
            ],
            True,
            7,
            None,
            [0, 0],
            [-15, -8],
            [-14, -7],
            [-8, -1],
        ),
        (
            None,
            [
                FoldMask(
                    first_train_timestamp=pd.Timestamp("2020-01-01 1:00"),
                    last_train_timestamp=pd.Timestamp("2020-01-31 10:00"),
                    target_timestamps=[pd.Timestamp("2020-01-31 14:00")],
                ),
                FoldMask(
                    first_train_timestamp=pd.Timestamp("2020-01-01 8:00"),
                    last_train_timestamp=pd.Timestamp("2020-01-31 17:00"),
                    target_timestamps=[pd.Timestamp("2020-01-31 21:00")],
                ),
            ],
            True,
            7,
            None,
            [1, 8],
            [-15, -8],
            [-14, -7],
            [-8, -1],
        ),
        (
            None,
            [
                FoldMask(
                    first_train_timestamp=None,
                    last_train_timestamp=pd.Timestamp("2020-01-30 20:00"),
                    target_timestamps=[pd.Timestamp("2020-01-31 00:00")],
                ),
                FoldMask(
                    first_train_timestamp=None,
                    last_train_timestamp=pd.Timestamp("2020-01-31 03:00"),
                    target_timestamps=[pd.Timestamp("2020-01-31 07:00")],
                ),
                FoldMask(
                    first_train_timestamp=None,
                    last_train_timestamp=pd.Timestamp("2020-01-31 10:00"),
                    target_timestamps=[pd.Timestamp("2020-01-31 14:00")],
                ),
                FoldMask(
                    first_train_timestamp=None,
                    last_train_timestamp=pd.Timestamp("2020-01-31 17:00"),
                    target_timestamps=[pd.Timestamp("2020-01-31 21:00")],
                ),
            ],
            2,
            7,
            None,
            [0, 0, 0, 0],
            [-29, -29, -15, -15],
            [-28, -21, -14, -7],
            [-22, -15, -8, -1],
        ),
    ],
)
def test_backtest_fold_info_timestamps(
    mode,
    n_folds,
    refit,
    horizon,
    stride,
    expected_train_starts,
    expected_train_ends,
    expected_test_starts,
    expected_test_ends,
    example_tsdf,
):
    """Check that Pipeline.backtest returns info dataframe with correct timestamps."""
    pipeline = Pipeline(model=NaiveModel(lag=horizon), horizon=horizon)
    _, _, info_df = pipeline.backtest(
        ts=example_tsdf, metrics=DEFAULT_METRICS, mode=mode, n_folds=n_folds, refit=refit, stride=stride
    )
    timestamp = example_tsdf.index

    np.testing.assert_array_equal(info_df["train_start_time"], timestamp[expected_train_starts])
    np.testing.assert_array_equal(info_df["train_end_time"], timestamp[expected_train_ends])
    np.testing.assert_array_equal(info_df["test_start_time"], timestamp[expected_test_starts])
    np.testing.assert_array_equal(info_df["test_end_time"], timestamp[expected_test_ends])


def test_backtest_refit_success(catboost_pipeline: Pipeline, big_example_tsdf: TSDataset):
    """Check that backtest without refit works on pipeline that supports it."""
    _ = catboost_pipeline.backtest(ts=big_example_tsdf, n_jobs=1, metrics=DEFAULT_METRICS, n_folds=3, refit=False)


def test_backtest_refit_fail(big_example_tsdf: TSDataset):
    """Check that backtest without refit doesn't work on pipeline that doesn't support it."""
    pipeline = Pipeline(
        model=NaiveModel(lag=7),
        transforms=[DifferencingTransform(in_column="target", inplace=True)],
        horizon=7,
    )
    with pytest.raises(ValueError, match="Test should go after the train without gaps"):
        _ = pipeline.backtest(ts=big_example_tsdf, n_jobs=1, metrics=DEFAULT_METRICS, n_folds=3, refit=False)


@pytest.mark.long_1
@pytest.mark.parametrize("refit", [True, False, 2])
def test_backtest_with_n_jobs(refit, catboost_pipeline: Pipeline, big_example_tsdf: TSDataset):
    """Check that Pipeline.backtest gives the same results in case of single and multiple jobs modes."""
    ts1 = deepcopy(big_example_tsdf)
    ts2 = deepcopy(big_example_tsdf)
    pipeline_1 = deepcopy(catboost_pipeline)
    pipeline_2 = deepcopy(catboost_pipeline)
    _, forecast_1, _ = pipeline_1.backtest(ts=ts1, n_jobs=1, n_folds=4, metrics=DEFAULT_METRICS, refit=refit)
    _, forecast_2, _ = pipeline_2.backtest(ts=ts2, n_jobs=3, n_folds=4, metrics=DEFAULT_METRICS, refit=refit)
    assert (forecast_1 == forecast_2).all().all()


def test_backtest_forecasts_sanity(step_ts: TSDataset):
    """Check that Pipeline.backtest gives correct forecasts according to the simple case."""
    ts, expected_metrics_df, expected_forecast_df = step_ts
    pipeline = Pipeline(model=NaiveModel(), horizon=5)
    metrics_df, forecast_df, _ = pipeline.backtest(ts, metrics=[MAE()], n_folds=3)

    assert np.all(metrics_df.reset_index(drop=True) == expected_metrics_df)
    assert np.all(forecast_df == expected_forecast_df)


def test_forecast_raise_error_if_no_ts():
    """Test that Pipeline raises error when calling forecast without ts."""
    pipeline = Pipeline(model=NaiveModel(), horizon=5)
    with pytest.raises(ValueError, match="There is no ts to forecast!"):
        _ = pipeline.forecast()


def test_forecast_pipeline_with_nan_at_the_end(ts_with_nans_in_tails):
    """Test that Pipeline can forecast with datasets with nans at the end."""
    pipeline = Pipeline(model=NaiveModel(), transforms=[TimeSeriesImputerTransform(strategy="forward_fill")], horizon=5)
    pipeline.fit(ts_with_nans_in_tails)
    forecast = pipeline.forecast()
    assert len(forecast.df) == 5


@pytest.mark.parametrize(
    "n_folds, horizon, stride, mode, expected_masks",
    (
        (
            2,
            3,
            3,
            CrossValidationMode.expand,
            [
                FoldMask(
                    first_train_timestamp="2020-01-01",
                    last_train_timestamp="2020-04-03",
                    target_timestamps=["2020-04-04", "2020-04-05", "2020-04-06"],
                ),
                FoldMask(
                    first_train_timestamp="2020-01-01",
                    last_train_timestamp="2020-04-06",
                    target_timestamps=["2020-04-07", "2020-04-08", "2020-04-09"],
                ),
            ],
        ),
        (
            2,
            3,
            1,
            CrossValidationMode.expand,
            [
                FoldMask(
                    first_train_timestamp="2020-01-01",
                    last_train_timestamp="2020-04-05",
                    target_timestamps=["2020-04-06", "2020-04-07", "2020-04-08"],
                ),
                FoldMask(
                    first_train_timestamp="2020-01-01",
                    last_train_timestamp="2020-04-06",
                    target_timestamps=["2020-04-07", "2020-04-08", "2020-04-09"],
                ),
            ],
        ),
        (
            2,
            3,
            5,
            CrossValidationMode.expand,
            [
                FoldMask(
                    first_train_timestamp="2020-01-01",
                    last_train_timestamp="2020-04-01",
                    target_timestamps=["2020-04-02", "2020-04-03", "2020-04-04"],
                ),
                FoldMask(
                    first_train_timestamp="2020-01-01",
                    last_train_timestamp="2020-04-06",
                    target_timestamps=["2020-04-07", "2020-04-08", "2020-04-09"],
                ),
            ],
        ),
        (
            2,
            3,
            3,
            CrossValidationMode.constant,
            [
                FoldMask(
                    first_train_timestamp="2020-01-01",
                    last_train_timestamp="2020-04-03",
                    target_timestamps=["2020-04-04", "2020-04-05", "2020-04-06"],
                ),
                FoldMask(
                    first_train_timestamp="2020-01-04",
                    last_train_timestamp="2020-04-06",
                    target_timestamps=["2020-04-07", "2020-04-08", "2020-04-09"],
                ),
            ],
        ),
        (
            2,
            3,
            1,
            CrossValidationMode.constant,
            [
                FoldMask(
                    first_train_timestamp="2020-01-01",
                    last_train_timestamp="2020-04-05",
                    target_timestamps=["2020-04-06", "2020-04-07", "2020-04-08"],
                ),
                FoldMask(
                    first_train_timestamp="2020-01-02",
                    last_train_timestamp="2020-04-06",
                    target_timestamps=["2020-04-07", "2020-04-08", "2020-04-09"],
                ),
            ],
        ),
        (
            2,
            3,
            5,
            CrossValidationMode.constant,
            [
                FoldMask(
                    first_train_timestamp="2020-01-01",
                    last_train_timestamp="2020-04-01",
                    target_timestamps=["2020-04-02", "2020-04-03", "2020-04-04"],
                ),
                FoldMask(
                    first_train_timestamp="2020-01-06",
                    last_train_timestamp="2020-04-06",
                    target_timestamps=["2020-04-07", "2020-04-08", "2020-04-09"],
                ),
            ],
        ),
    ),
)
def test_generate_masks_from_n_folds(example_tsds: TSDataset, n_folds, horizon, stride, mode, expected_masks):
    masks = Pipeline._generate_masks_from_n_folds(
        ts=example_tsds, n_folds=n_folds, horizon=horizon, stride=stride, mode=mode
    )
    for mask, expected_mask in zip(masks, expected_masks):
        assert mask.first_train_timestamp == expected_mask.first_train_timestamp
        assert mask.last_train_timestamp == expected_mask.last_train_timestamp
        assert mask.target_timestamps == expected_mask.target_timestamps


@pytest.mark.parametrize(
    "mask", (FoldMask("2020-01-01", "2020-01-02", ["2020-01-03"]), FoldMask("2020-01-03", "2020-01-05", ["2020-01-06"]))
)
@pytest.mark.parametrize(
    "ts_name", ["simple_ts", "simple_ts_starting_with_nans_one_segment", "simple_ts_starting_with_nans_all_segments"]
)
def test_generate_folds_datasets(ts_name, mask, request):
    """Check _generate_folds_datasets for correct work."""
    ts = request.getfixturevalue(ts_name)
    pipeline = Pipeline(model=NaiveModel(lag=7))
    mask = pipeline._prepare_fold_masks(ts=ts, masks=[mask], mode=CrossValidationMode.expand, stride=-1)[0]
    train, test = list(pipeline._generate_folds_datasets(ts, [mask], 4))[0]
    assert train.index.min() == np.datetime64(mask.first_train_timestamp)
    assert train.index.max() == np.datetime64(mask.last_train_timestamp)
    assert test.index.min() == np.datetime64(mask.last_train_timestamp) + np.timedelta64(1, "D")
    assert test.index.max() == np.datetime64(mask.last_train_timestamp) + np.timedelta64(4, "D")


@pytest.mark.parametrize(
    "mask", (FoldMask(None, "2020-01-02", ["2020-01-03"]), FoldMask(None, "2020-01-05", ["2020-01-06"]))
)
@pytest.mark.parametrize(
    "ts_name", ["simple_ts", "simple_ts_starting_with_nans_one_segment", "simple_ts_starting_with_nans_all_segments"]
)
def test_generate_folds_datasets_without_first_date(ts_name, mask, request):
    """Check _generate_folds_datasets for correct work without first date."""
    ts = request.getfixturevalue(ts_name)
    pipeline = Pipeline(model=NaiveModel(lag=7))
    mask = pipeline._prepare_fold_masks(ts=ts, masks=[mask], mode=CrossValidationMode.expand, stride=-1)[0]
    train, test = list(pipeline._generate_folds_datasets(ts, [mask], 4))[0]
    assert train.index.min() == np.datetime64(ts.index.min())
    assert train.index.max() == np.datetime64(mask.last_train_timestamp)
    assert test.index.min() == np.datetime64(mask.last_train_timestamp) + np.timedelta64(1, "D")
    assert test.index.max() == np.datetime64(mask.last_train_timestamp) + np.timedelta64(4, "D")


@pytest.mark.parametrize(
    "mask,expected",
    (
        (FoldMask("2020-01-01", "2020-01-07", ["2020-01-10"]), {"segment_0": 0, "segment_1": 11}),
        (FoldMask("2020-01-01", "2020-01-07", ["2020-01-08", "2020-01-11"]), {"segment_0": 95.5, "segment_1": 5}),
    ),
)
def test_process_fold_forecast(ts_process_fold_forecast, mask: FoldMask, expected: Dict[str, List[float]]):
    train, test = ts_process_fold_forecast.train_test_split(
        train_start=mask.first_train_timestamp, train_end=mask.last_train_timestamp
    )

    pipeline = Pipeline(model=NaiveModel(lag=5), transforms=[], horizon=4)
    pipeline = pipeline.fit(ts=train)
    forecast = pipeline.forecast()
    fold = pipeline._process_fold_forecast(
        forecast=forecast, train=train, test=test, pipeline=pipeline, fold_number=1, mask=mask, metrics=[MAE()]
    )
    for seg in fold["metrics"]["MAE"].keys():
        assert fold["metrics"]["MAE"][seg] == expected[seg]


def test_make_backtest_fold_groups_refit_true():
    masks = [MagicMock() for _ in range(2)]
    obtained_results = Pipeline._make_backtest_fold_groups(masks=masks, refit=True)
    expected_results = [
        {
            "train_fold_number": 0,
            "train_mask": masks[0],
            "forecast_fold_numbers": [0],
            "forecast_masks": [masks[0]],
        },
        {
            "train_fold_number": 1,
            "train_mask": masks[1],
            "forecast_fold_numbers": [1],
            "forecast_masks": [masks[1]],
        },
    ]
    assert obtained_results == expected_results


def test_make_backtest_fold_groups_refit_false():
    masks = [MagicMock() for _ in range(2)]
    obtained_results = Pipeline._make_backtest_fold_groups(masks=masks, refit=False)
    expected_results = [
        {
            "train_fold_number": 0,
            "train_mask": masks[0],
            "forecast_fold_numbers": [0, 1],
            "forecast_masks": [masks[0], masks[1]],
        }
    ]
    assert obtained_results == expected_results


def test_make_backtest_fold_groups_refit_int():
    masks = [MagicMock() for _ in range(5)]
    obtained_results = Pipeline._make_backtest_fold_groups(masks=masks, refit=2)
    expected_results = [
        {
            "train_fold_number": 0,
            "train_mask": masks[0],
            "forecast_fold_numbers": [0, 1],
            "forecast_masks": [masks[0], masks[1]],
        },
        {
            "train_fold_number": 2,
            "train_mask": masks[2],
            "forecast_fold_numbers": [2, 3],
            "forecast_masks": [masks[2], masks[3]],
        },
        {
            "train_fold_number": 4,
            "train_mask": masks[4],
            "forecast_fold_numbers": [4],
            "forecast_masks": [masks[4]],
        },
    ]
    assert obtained_results == expected_results


@pytest.mark.parametrize(
    "n_folds, refit, expected_refits",
    [
        (1, 1, 1),
        (1, 2, 1),
        (3, 1, 3),
        (3, 2, 2),
        (3, 3, 1),
        (3, 4, 1),
        (4, 1, 4),
        (4, 2, 2),
        (4, 3, 2),
        (4, 4, 1),
        (4, 5, 1),
    ],
)
def test_make_backtest_fold_groups_length_refit_int(n_folds, refit, expected_refits):
    masks = [MagicMock() for _ in range(n_folds)]
    obtained_results = Pipeline._make_backtest_fold_groups(masks=masks, refit=refit)
    assert len(obtained_results) == expected_refits


@pytest.mark.parametrize(
    "lag,expected", ((5, {"segment_0": 76.923077, "segment_1": 90.909091}), (6, {"segment_0": 100, "segment_1": 120}))
)
def test_backtest_one_point(simple_ts: TSDataset, lag: int, expected: Dict[str, List[float]]):
    mask = FoldMask(
        simple_ts.index.min(),
        simple_ts.index.min() + np.timedelta64(6, "D"),
        [simple_ts.index.min() + np.timedelta64(8, "D")],
    )
    pipeline = Pipeline(model=NaiveModel(lag=lag), transforms=[], horizon=2)
    metrics_df, _, _ = pipeline.backtest(ts=simple_ts, metrics=[SMAPE()], n_folds=[mask], aggregate_metrics=True)
    metrics = dict(metrics_df.values)
    for segment in expected.keys():
        assert segment in metrics.keys()
        np.testing.assert_array_almost_equal(expected[segment], metrics[segment])


@pytest.mark.parametrize(
    "lag,expected", ((4, {"segment_0": 0, "segment_1": 0}), (7, {"segment_0": 0, "segment_1": 0.5}))
)
def test_backtest_two_points(masked_ts: TSDataset, lag: int, expected: Dict[str, List[float]]):
    mask = FoldMask(
        masked_ts.index.min(),
        masked_ts.index.min() + np.timedelta64(6, "D"),
        [masked_ts.index.min() + np.timedelta64(9, "D"), masked_ts.index.min() + np.timedelta64(10, "D")],
    )
    pipeline = Pipeline(model=NaiveModel(lag=lag), transforms=[], horizon=4)
    metrics_df, _, _ = pipeline.backtest(ts=masked_ts, metrics=[MAE()], n_folds=[mask], aggregate_metrics=True)
    metrics = dict(metrics_df.values)
    for segment in expected.keys():
        assert segment in metrics.keys()
        np.testing.assert_array_almost_equal(expected[segment], metrics[segment])


def test_sanity_backtest_naive_with_intervals(weekly_period_ts):
    train_ts, _ = weekly_period_ts
    quantiles = (0.01, 0.99)
    pipeline = Pipeline(model=NaiveModel(), horizon=5)
    _, forecast_df, _ = pipeline.backtest(
        ts=train_ts,
        metrics=[MAE(), Width(quantiles=quantiles)],
        forecast_params={"quantiles": quantiles, "prediction_interval": True},
    )
    features = forecast_df.columns.get_level_values(1)
    assert f"target_{quantiles[0]}" in features
    assert f"target_{quantiles[1]}" in features


@pytest.mark.long_1
def test_backtest_pass_with_filter_transform(ts_with_feature):
    ts = ts_with_feature

    pipeline = Pipeline(
        model=ProphetModel(),
        transforms=[
            LogTransform(in_column="feature_1"),
            FilterFeaturesTransform(exclude=["feature_1"], return_features=True),
        ],
        horizon=10,
    )
    pipeline.backtest(ts=ts, metrics=[MAE()], aggregate_metrics=True)


@pytest.mark.parametrize(
    "ts_name", ["simple_ts_starting_with_nans_one_segment", "simple_ts_starting_with_nans_all_segments"]
)
def test_backtest_nans_at_beginning(ts_name, request):
    ts = request.getfixturevalue(ts_name)
    pipeline = Pipeline(model=NaiveModel(), horizon=2)
    _ = pipeline.backtest(
        ts=ts,
        metrics=[MAE()],
        n_folds=2,
    )


@pytest.mark.parametrize(
    "ts_name", ["simple_ts_starting_with_nans_one_segment", "simple_ts_starting_with_nans_all_segments"]
)
def test_backtest_nans_at_beginning_with_mask(ts_name, request):
    ts = request.getfixturevalue(ts_name)
    mask = FoldMask(
        ts.index.min(),
        ts.index.min() + np.timedelta64(5, "D"),
        [ts.index.min() + np.timedelta64(6, "D"), ts.index.min() + np.timedelta64(8, "D")],
    )
    pipeline = Pipeline(model=NaiveModel(), horizon=3)
    _ = pipeline.backtest(
        ts=ts,
        metrics=[MAE()],
        n_folds=[mask],
    )


def test_forecast_backtest_correct_ordering(step_ts: TSDataset):
    ts, _, expected_forecast_df = step_ts
    pipeline = Pipeline(model=NaiveModel(), horizon=5)
    _, forecast_df, _ = pipeline.backtest(ts=ts, metrics=[MAE()], n_folds=3)
    assert np.all(forecast_df.values == expected_forecast_df.values)


def test_pipeline_with_deepmodel(example_tsds):
    from etna.models.nn import RNNModel

    pipeline = Pipeline(
        model=RNNModel(input_size=1, encoder_length=14, decoder_length=14, trainer_params=dict(max_epochs=1)),
        transforms=[],
        horizon=2,
    )
    _ = pipeline.backtest(ts=example_tsds, metrics=[MAE()], n_folds=2, aggregate_metrics=True)


@pytest.mark.parametrize(
    "model, transforms",
    [
        (
            CatBoostMultiSegmentModel(iterations=100),
            [DateFlagsTransform(), LagTransform(in_column="target", lags=list(range(7, 15)))],
        ),
        (
            LinearPerSegmentModel(),
            [DateFlagsTransform(), LagTransform(in_column="target", lags=list(range(7, 15)))],
        ),
        (SeasonalMovingAverageModel(window=2, seasonality=7), []),
        (SARIMAXModel(), []),
        (ProphetModel(), []),
    ],
)
def test_predict_format(model, transforms, example_tsds):
    ts = example_tsds
    pipeline = Pipeline(model=model, transforms=transforms, horizon=7)
    pipeline.fit(ts)

    start_idx = 50
    end_idx = 70
    start_timestamp = ts.index[start_idx]
    end_timestamp = ts.index[end_idx]
    num_points = end_idx - start_idx + 1

    # create a separate TSDataset with slice of original timestamps
    predict_ts = deepcopy(ts)
    predict_ts.df = predict_ts.df.iloc[5 : end_idx + 5]

    result_ts = pipeline.predict(ts=predict_ts, start_timestamp=start_timestamp, end_timestamp=end_timestamp)
    result_df = result_ts.to_pandas(flatten=True)

    assert not np.any(result_df["target"].isna())
    assert len(result_df) == len(example_tsds.segments) * num_points


def test_predict_values(example_tsds):
    original_ts = deepcopy(example_tsds)

    model = LinearPerSegmentModel()
    transforms = [AddConstTransform(in_column="target", value=10, inplace=True), DateFlagsTransform()]
    pipeline = Pipeline(model=model, transforms=transforms, horizon=5)
    pipeline.fit(example_tsds)
    predictions_pipeline = pipeline.predict(ts=original_ts)

    original_ts.fit_transform(transforms)
    model.fit(original_ts)
    predictions_manual = model.predict(original_ts)
    predictions_manual.inverse_transform(transforms)

    pd.testing.assert_frame_equal(predictions_pipeline.to_pandas(), predictions_manual.to_pandas())


@pytest.mark.parametrize("load_ts", [True, False])
@pytest.mark.parametrize(
    "model, transforms",
    [
        (
            CatBoostMultiSegmentModel(iterations=100),
            [DateFlagsTransform(), LagTransform(in_column="target", lags=list(range(3, 10)))],
        ),
        (
            LinearPerSegmentModel(),
            [DateFlagsTransform(), LagTransform(in_column="target", lags=list(range(3, 10)))],
        ),
        (SeasonalMovingAverageModel(window=2, seasonality=7), []),
        (SARIMAXModel(), []),
        (ProphetModel(), []),
    ],
)
def test_save_load(load_ts, model, transforms, example_tsds):
    horizon = 3
    pipeline = Pipeline(model=model, transforms=transforms, horizon=horizon)
    assert_pipeline_equals_loaded_original(pipeline=pipeline, ts=example_tsds, load_ts=load_ts)


@pytest.mark.parametrize(
    "model, transforms",
    [
        (
            CatBoostMultiSegmentModel(iterations=100),
            [DateFlagsTransform(), LagTransform(in_column="target", lags=list(range(3, 10)))],
        ),
        (
            LinearPerSegmentModel(),
            [DateFlagsTransform(), LagTransform(in_column="target", lags=list(range(3, 10)))],
        ),
        (SeasonalMovingAverageModel(window=2, seasonality=7), []),
        (SARIMAXModel(), []),
        (ProphetModel(), []),
    ],
)
def test_forecast_given_ts(model, transforms, example_tsds):
    horizon = 3
    pipeline = Pipeline(model=model, transforms=transforms, horizon=horizon)
    assert_pipeline_forecasts_given_ts(pipeline=pipeline, ts=example_tsds, horizon=horizon)


@pytest.mark.parametrize(
    "model, transforms",
    [
        (
            CatBoostMultiSegmentModel(iterations=100),
            [DateFlagsTransform(), LagTransform(in_column="target", lags=list(range(3, 10)))],
        ),
        (
            LinearPerSegmentModel(),
            [DateFlagsTransform(), LagTransform(in_column="target", lags=list(range(3, 10)))],
        ),
        (SeasonalMovingAverageModel(window=2, seasonality=7), []),
        (SARIMAXModel(), []),
        (ProphetModel(), []),
    ],
)
def test_forecast_given_ts_with_prediction_interval(model, transforms, example_tsds):
    horizon = 3
    pipeline = Pipeline(model=model, transforms=transforms, horizon=horizon)
    assert_pipeline_forecasts_given_ts_with_prediction_intervals(pipeline=pipeline, ts=example_tsds, horizon=horizon)


@pytest.mark.parametrize(
    "model_fixture",
    (
        "non_prediction_interval_context_ignorant_dummy_model",
        "non_prediction_interval_context_required_dummy_model",
        "prediction_interval_context_ignorant_dummy_model",
        "prediction_interval_context_required_dummy_model",
    ),
)
def test_predict_return_components(
    example_tsds, model_fixture, request, expected_component_a=20, expected_component_b=180
):
    model = request.getfixturevalue(model_fixture)
    pipeline = Pipeline(model=model)
    pipeline.fit(example_tsds)
    forecast = pipeline.predict(ts=example_tsds, return_components=True)
    assert sorted(forecast.target_components_names) == sorted(["target_component_a", "target_component_b"])

    target_components_df = TSDataset.to_flatten(forecast.get_target_components())
    assert (target_components_df["target_component_a"] == expected_component_a).all()
    assert (target_components_df["target_component_b"] == expected_component_b).all()


@pytest.mark.parametrize(
    "model, transforms, expected_params_to_tune",
    [
        (
            CatBoostMultiSegmentModel(iterations=100),
            [DateFlagsTransform(), LagTransform(in_column="target", lags=list(range(3, 10)))],
            {
                "model.learning_rate": FloatDistribution(low=1e-4, high=0.5, log=True),
                "model.depth": IntDistribution(low=1, high=11, step=1),
                "model.l2_leaf_reg": FloatDistribution(low=0.1, high=200.0, log=True),
                "model.random_strength": FloatDistribution(low=1e-05, high=10.0, log=True),
                "transforms.0.day_number_in_week": CategoricalDistribution([False, True]),
                "transforms.0.day_number_in_month": CategoricalDistribution([False, True]),
                "transforms.0.day_number_in_year": CategoricalDistribution([False, True]),
                "transforms.0.week_number_in_month": CategoricalDistribution([False, True]),
                "transforms.0.week_number_in_year": CategoricalDistribution([False, True]),
                "transforms.0.month_number_in_year": CategoricalDistribution([False, True]),
                "transforms.0.season_number": CategoricalDistribution([False, True]),
                "transforms.0.year_number": CategoricalDistribution([False, True]),
                "transforms.0.is_weekend": CategoricalDistribution([False, True]),
            },
        ),
    ],
)
def test_params_to_tune(model, transforms, expected_params_to_tune):
    horizon = 3
    pipeline = Pipeline(model=model, transforms=transforms, horizon=horizon)

    obtained_params_to_tune = pipeline.params_to_tune()

    assert obtained_params_to_tune == expected_params_to_tune
