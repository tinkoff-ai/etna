from tempfile import NamedTemporaryFile

import pytest
from loguru import logger as _logger

from etna.models import LinearPerSegmentModel, LinearMultiSegmentModel
from etna.model_selection import TimeSeriesCrossValidation
from etna.models import CatBoostModelMultiSegment
from etna.metrics import MAE, MSE, SMAPE
from etna.transforms import LagTransform
from etna.transforms import AddConstTransform, DateFlagsTransform
from etna.datasets import TSDataset
from etna.loggers import ConsoleLogger
from etna.loggers import tslogger


def test_tsdataset_fit_transform_logging(example_tsds: TSDataset):
    """Check working of logging inside fit_transform of TSDataset."""
    transforms = [LagTransform(lags=5, in_column="target"), AddConstTransform(value=5, in_column="target")]
    file = NamedTemporaryFile()
    _logger.add(file.name)
    idx = tslogger.add(ConsoleLogger())
    example_tsds.fit_transform(transforms=transforms)
    with open(file.name, "r") as in_file:
        lines = in_file.readlines()
        assert len(lines) == len(transforms)
        for line, transform in zip(lines, transforms):
            assert transform.__class__.__name__ in line
    tslogger.remove(idx)


def test_backtest_logging(example_tsds: TSDataset):
    """Check working of logging inside backtest."""
    file = NamedTemporaryFile()
    _logger.add(file.name)
    idx = tslogger.add(ConsoleLogger())
    metrics = [MAE(), MSE(), SMAPE()]
    metrics_str = ["MAE", "MSE", "SMAPE"]
    date_flags = DateFlagsTransform(day_number_in_week=True, day_number_in_month=True)
    tsvc = TimeSeriesCrossValidation(model=CatBoostModelMultiSegment(), horizon=10, metrics=metrics, n_jobs=1)
    tsvc.backtest(ts=example_tsds, transforms=[date_flags])
    with open(file.name, "r") as in_file:
        lines = in_file.readlines()
        # remain lines only about backtest
        lines = [line for line in lines if "backtest" in line]
        assert len(lines) == len(metrics) * tsvc.n_folds * len(example_tsds.segments)
        assert all([any([metric_str in line for metric_str in metrics_str]) for line in lines])
    tslogger.remove(idx)


@pytest.mark.parametrize("model", [LinearPerSegmentModel(), LinearMultiSegmentModel()])
def test_model_logging(example_tsds, model):
    """Check working of logging in fit/forecast of model."""
    horizon = 7
    lags = LagTransform(in_column="target", lags=[i + horizon for i in range(1, 5 + 1)])
    example_tsds.fit_transform([lags])

    file = NamedTemporaryFile()
    _logger.add(file.name)
    idx = tslogger.add(ConsoleLogger())

    model.fit(example_tsds)
    to_forecast = example_tsds.make_future(horizon)
    model.forecast(to_forecast)

    with open(file.name, "r") as in_file:
        lines = in_file.readlines()
        assert len(lines) == 2
        assert "fit" in lines[0]
        assert "forecast" in lines[1]

    tslogger.remove(idx)
