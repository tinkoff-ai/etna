from copy import deepcopy

import pytest

from etna.commands.backtest_command import ADDITIONAL_BACKTEST_PARAMETERS
from etna.commands.forecast_command import ADDITIONAL_FORECAST_PARAMETERS
from etna.commands.forecast_command import ADDITIONAL_PIPELINE_PARAMETERS
from etna.commands.utils import _estimate_n_folds
from etna.commands.utils import _max_n_folds_backtest
from etna.commands.utils import _max_n_folds_forecast
from etna.commands.utils import estimate_max_n_folds
from etna.commands.utils import remove_params
from etna.metrics import MAE
from etna.models import HoltWintersModel
from etna.models import LinearPerSegmentModel
from etna.models import SeasonalMovingAverageModel
from etna.pipeline import Pipeline
from etna.transforms import DensityOutliersTransform
from etna.transforms import DifferencingTransform
from etna.transforms import LagTransform
from etna.transforms import MeanTransform


def run_estimate_max_n_folds_forecast_test(pipeline, context_size, ts, expected):
    pipeline.fit(ts=ts)

    n_folds = estimate_max_n_folds(pipeline=pipeline, method_name="forecast", context_size=context_size)

    assert n_folds == expected
    pipeline.forecast(prediction_interval=True, n_folds=n_folds)


def run_estimate_max_n_folds_backtest_test(pipeline, context_size, ts, stride, expected):
    n_folds = estimate_max_n_folds(
        pipeline=pipeline, ts=ts, method_name="backtest", stride=stride, context_size=context_size
    )

    assert n_folds == expected
    pipeline.backtest(ts=ts, metrics=[MAE()], n_folds=n_folds, stride=stride)


@pytest.fixture
def pipeline_with_context(request):
    if hasattr(request, "param"):
        horizon = request.param["horizon"]
        window = request.param["window"]
    else:
        horizon = 1
        window = 1

    pipeline = Pipeline(transforms=[], model=SeasonalMovingAverageModel(seasonality=1, window=window), horizon=horizon)
    return pipeline


@pytest.fixture
def pipeline_without_context(request):
    horizon = request.param if hasattr(request, "param") else 1
    pipeline = Pipeline(transforms=[], model=HoltWintersModel(), horizon=horizon)
    return pipeline


@pytest.fixture
def pipeline_with_transforms():
    transforms = [
        LagTransform(in_column="target", lags=[14, 17]),
        DifferencingTransform(in_column="target"),
        MeanTransform(in_column="target", window=7),
        DensityOutliersTransform(in_column="target"),
    ]

    pipeline = Pipeline(transforms=transforms, model=LinearPerSegmentModel(), horizon=14)
    return pipeline


@pytest.mark.parametrize(
    "num_points, horizon, stride, context_size, expected",
    (
        (13, 2, 2, 2, 5),
        (13, 2, 1, 2, 10),
        (13, 2, 2, 1, 6),
        (13, 2, 1, 1, 11),
        (13, 1, 1, 1, 12),
        (13, 4, 4, 6, 1),
        (13, 4, 1, 6, 4),
        (10, 5, 1, 5, 1),
        (10, 5, 5, 5, 1),
    ),
)
def test_private_estimate_n_folds(num_points, horizon, stride, context_size, expected):
    res = _estimate_n_folds(num_points=num_points, horizon=horizon, stride=stride, context_size=context_size)
    assert res == expected


def test_estimate_n_folds_not_enough_points(num_points=10, horizon=7, stride=1, context_size=5):
    with pytest.raises(ValueError, match="Not enough data points!"):
        _ = _estimate_n_folds(num_points=num_points, horizon=horizon, stride=stride, context_size=context_size)


def test_estimate_n_folds_forecast_no_ts(pipeline_without_context):
    with pytest.raises(ValueError, match="There is no ts for forecast method!"):
        _ = _max_n_folds_forecast(pipeline=pipeline_without_context, ts=None, context_size=1)


def test_estimate_n_folds_backtest_no_ts(pipeline_without_context):
    with pytest.raises(ValueError, match="Parameter `ts` is required when estimating for backtest method"):
        _ = estimate_max_n_folds(pipeline=pipeline_without_context, method_name="backtest", context_size=1)


def test_estimate_n_folds_backtest_intervals_error(pipeline_without_context, example_tsds):
    with pytest.raises(
        NotImplementedError, match="Number of folds estimation for backtest with intervals is not implemented!"
    ):
        _ = _max_n_folds_backtest(
            pipeline=pipeline_without_context,
            ts=example_tsds,
            forecast_params={"prediction_interval": True},
            context_size=1,
        )


def test_estimate_max_n_folds_invalid_method_name(pipeline_without_context, example_tsds, method_name="fit"):
    with pytest.raises(ValueError, match="fit is not a valid method name."):
        _ = estimate_max_n_folds(
            pipeline=pipeline_without_context, ts=example_tsds, method_name=method_name, context_size=1
        )


def test_estimate_max_n_folds_empty_ts(pipeline_without_context, empty_ts):
    with pytest.raises(ValueError, match="Not enough data points!"):
        _ = estimate_max_n_folds(pipeline=pipeline_without_context, ts=empty_ts, method_name="forecast", context_size=1)


def test_estimate_max_n_folds_negative_context(pipeline_without_context, example_tsds):
    with pytest.raises(ValueError, match="Pipeline `context_size` parameter must be positive integer!"):
        _ = estimate_max_n_folds(
            pipeline=pipeline_without_context, ts=example_tsds, method_name="forecast", context_size=-1
        )


def test_estimate_max_n_folds_forecast_with_ts(pipeline_without_context, example_tsds, context_size=3, expected=7):
    pipeline = pipeline_without_context

    pipeline.fit(ts=example_tsds)

    ts_to_forecast = deepcopy(example_tsds)
    ts_to_forecast.df = ts_to_forecast.df.iloc[-(context_size + expected) :]

    n_folds = estimate_max_n_folds(
        pipeline=pipeline, method_name="forecast", ts=ts_to_forecast, context_size=context_size
    )

    assert n_folds == expected
    pipeline.forecast(ts=ts_to_forecast, prediction_interval=True, n_folds=n_folds)


@pytest.mark.parametrize(
    "pipeline_without_context,context_size,ts_name,expected",
    (
        (1, 3, "example_tsds", 97),
        (4, 3, "example_tsds", 24),
        (13, 3, "example_tsds", 7),
        (97, 3, "example_tsds", 1),
        (40, 3, "ts_with_different_series_length", 18),
    ),
    indirect=["pipeline_without_context"],
)
def test_estimate_max_n_folds_forecast_no_context(pipeline_without_context, context_size, ts_name, expected, request):
    ts = request.getfixturevalue(ts_name)
    run_estimate_max_n_folds_forecast_test(
        pipeline=pipeline_without_context, ts=ts, expected=expected, context_size=context_size
    )


@pytest.mark.parametrize(
    "pipeline_with_context,context_size,ts_name,expected",
    (
        ({"horizon": 1, "window": 1}, 1, "example_tsds", 99),
        ({"horizon": 1, "window": 2}, 2, "example_tsds", 98),
        ({"horizon": 13, "window": 10}, 10, "example_tsds", 6),
        ({"horizon": 10, "window": 1}, 1, "ts_with_different_series_length", 74),
    ),
    indirect=["pipeline_with_context"],
)
def test_estimate_max_n_folds_forecast_with_context(pipeline_with_context, context_size, ts_name, expected, request):
    ts = request.getfixturevalue(ts_name)
    run_estimate_max_n_folds_forecast_test(
        pipeline=pipeline_with_context, context_size=context_size, ts=ts, expected=expected
    )


@pytest.mark.parametrize(
    "context_size,ts_name,expected",
    (
        (18, "example_tsds", 5),
        (18, "ts_with_different_series_length", 51),
    ),
)
def test_estimate_max_n_folds_forecast_with_transforms(
    pipeline_with_transforms, context_size, ts_name, expected, request
):
    ts = request.getfixturevalue(ts_name)
    run_estimate_max_n_folds_forecast_test(
        pipeline=pipeline_with_transforms, ts=ts, expected=expected, context_size=context_size
    )


@pytest.mark.parametrize(
    "pipeline_without_context,context_size,stride,ts_name,expected",
    (
        (4, 3, 8, "example_tsds", 12),
        (13, 3, 13, "example_tsds", 7),
        (13, 3, 3, "example_tsds", 29),
        (97, 3, 3, "example_tsds", 1),
        (40, 3, 60, "ts_with_different_series_length", 12),
    ),
    indirect=["pipeline_without_context"],
)
def test_estimate_max_n_folds_backtest_no_context(
    pipeline_without_context, context_size, stride, ts_name, expected, request
):
    ts = request.getfixturevalue(ts_name)
    run_estimate_max_n_folds_backtest_test(
        pipeline=pipeline_without_context, context_size=context_size, ts=ts, stride=stride, expected=expected
    )


@pytest.mark.parametrize(
    "pipeline_with_context,context_size,stride,ts_name,expected",
    (
        ({"horizon": 1, "window": 1}, 1, 8, "example_tsds", 13),
        ({"horizon": 5, "window": 8}, 8, 13, "example_tsds", 7),
        ({"horizon": 13, "window": 7}, 7, 3, "example_tsds", 27),
        ({"horizon": 13, "window": 60}, 60, 40, "ts_with_different_series_length", 17),
    ),
    indirect=["pipeline_with_context"],
)
def test_estimate_max_n_folds_backtest_with_context(
    pipeline_with_context, context_size, stride, ts_name, expected, request
):
    ts = request.getfixturevalue(ts_name)
    run_estimate_max_n_folds_backtest_test(
        pipeline=pipeline_with_context, context_size=context_size, ts=ts, stride=stride, expected=expected
    )


@pytest.mark.parametrize(
    "context_size,stride,ts_name,expected",
    (
        (18, 1, "example_tsds", 69),
        (18, 14, "example_tsds", 5),
        (18, 60, "ts_with_different_series_length", 12),
    ),
)
def test_estimate_max_n_folds_backtest_with_transforms(
    pipeline_with_transforms, context_size, stride, ts_name, expected, request
):
    ts = request.getfixturevalue(ts_name)
    run_estimate_max_n_folds_backtest_test(
        pipeline=pipeline_with_transforms, context_size=context_size, ts=ts, stride=stride, expected=expected
    )


@pytest.mark.parametrize(
    "params,to_remove,expected",
    (
        ({"start_timestamp": "2021-09-10"}, ADDITIONAL_FORECAST_PARAMETERS, {}),
        (
            {"prediction_interval": True, "n_folds": 3, "start_timestamp": "2021-09-10"},
            ADDITIONAL_FORECAST_PARAMETERS,
            {"prediction_interval": True, "n_folds": 3},
        ),
        (
            {"prediction_interval": True, "n_folds": 3, "quantiles": [0.025, 0.975]},
            ADDITIONAL_FORECAST_PARAMETERS,
            {"prediction_interval": True, "n_folds": 3, "quantiles": [0.025, 0.975]},
        ),
        (
            {"prediction_interval": True, "estimate_n_folds": True, "start_timestamp": "2021-09-10"},
            ADDITIONAL_FORECAST_PARAMETERS,
            {"prediction_interval": True},
        ),
        (
            {"n_folds": 2, "n_jobs": 4, "estimate_n_folds": True},
            ADDITIONAL_BACKTEST_PARAMETERS,
            {"n_folds": 2, "n_jobs": 4},
        ),
        (
            {"_target_": "etna.pipeline.Pipeline", "horizon": 4, "context_size": 1},
            ADDITIONAL_PIPELINE_PARAMETERS,
            {"_target_": "etna.pipeline.Pipeline", "horizon": 4},
        ),
    ),
)
def test_remove_params(params, to_remove, expected):
    result = remove_params(params=params, to_remove=to_remove)
    assert result == expected
