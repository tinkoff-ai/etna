from tempfile import NamedTemporaryFile
from typing import Sequence

import pytest
from loguru import logger as _logger

from etna.datasets import TSDataset
from etna.loggers import ConsoleLogger
from etna.loggers import tslogger
from etna.metrics import MAE
from etna.metrics import MSE
from etna.metrics import SMAPE
from etna.metrics import Metric
from etna.models import CatBoostMultiSegmentModel
from etna.models import LinearMultiSegmentModel
from etna.models import LinearPerSegmentModel
from etna.models import ProphetModel
from etna.pipeline import Pipeline
from etna.transforms import AddConstTransform
from etna.transforms import DateFlagsTransform
from etna.transforms import LagTransform
from etna.transforms import Transform


def check_logged_transforms(log_file: str, transforms: Sequence[Transform]):
    """Check that transforms are logged into the file."""
    with open(log_file, "r") as in_file:
        lines = in_file.readlines()
        assert len(lines) == len(transforms)
        for line, transform in zip(lines, transforms):
            assert transform.__class__.__name__ in line


def test_tsdataset_transform_logging(example_tsds: TSDataset):
    """Check working of logging inside `TSDataset.transform`."""
    transforms = [LagTransform(lags=5, in_column="target"), AddConstTransform(value=5, in_column="target")]
    file = NamedTemporaryFile()
    _logger.add(file.name)
    example_tsds.fit_transform(transforms=transforms)
    idx = tslogger.add(ConsoleLogger())
    example_tsds.transform(transforms=example_tsds.transforms)
    check_logged_transforms(log_file=file.name, transforms=transforms)
    tslogger.remove(idx)


def test_tsdataset_fit_transform_logging(example_tsds: TSDataset):
    """Check working of logging inside `TSDataset.fit_transform`."""
    transforms = [LagTransform(lags=5, in_column="target"), AddConstTransform(value=5, in_column="target")]
    file = NamedTemporaryFile()
    _logger.add(file.name)
    idx = tslogger.add(ConsoleLogger())
    example_tsds.fit_transform(transforms=transforms)
    check_logged_transforms(log_file=file.name, transforms=transforms)
    tslogger.remove(idx)


def test_tsdataset_make_future_logging(example_tsds: TSDataset):
    """Check working of logging inside `TSDataset.make_future`."""
    transforms = [LagTransform(lags=5, in_column="target"), AddConstTransform(value=5, in_column="target")]
    file = NamedTemporaryFile()
    _logger.add(file.name)
    example_tsds.fit_transform(transforms=transforms)
    idx = tslogger.add(ConsoleLogger())
    _ = example_tsds.make_future(5)
    check_logged_transforms(log_file=file.name, transforms=transforms)
    tslogger.remove(idx)


def test_tsdataset_inverse_transform_logging(example_tsds: TSDataset):
    """Check working of logging inside `TSDataset.inverse_transform`."""
    transforms = [LagTransform(lags=5, in_column="target"), AddConstTransform(value=5, in_column="target")]
    file = NamedTemporaryFile()
    _logger.add(file.name)
    example_tsds.fit_transform(transforms=transforms)
    idx = tslogger.add(ConsoleLogger())
    example_tsds.inverse_transform()
    check_logged_transforms(log_file=file.name, transforms=transforms[::-1])
    tslogger.remove(idx)


@pytest.mark.parametrize("metric", [MAE(), MSE(), MAE(mode="macro")])
def test_metric_logging(example_tsds: TSDataset, metric: Metric):
    """Check working of logging inside `Metric.__call__`."""
    file = NamedTemporaryFile()
    _logger.add(file.name)

    horizon = 10
    ts_train, ts_test = example_tsds.train_test_split(test_size=horizon)
    pipeline = Pipeline(model=ProphetModel(), horizon=horizon)
    pipeline.fit(ts_train)
    ts_forecast = pipeline.forecast()
    idx = tslogger.add(ConsoleLogger())
    _ = metric(y_true=ts_test, y_pred=ts_forecast)

    with open(file.name, "r") as in_file:
        lines = in_file.readlines()
        assert len(lines) == 1
        assert repr(metric) in lines[0]
    tslogger.remove(idx)


def test_backtest_logging(example_tsds: TSDataset):
    """Check working of logging inside backtest."""
    file = NamedTemporaryFile()
    _logger.add(file.name)
    idx = tslogger.add(ConsoleLogger())
    metrics = [MAE(), MSE(), SMAPE()]
    metrics_str = ["MAE", "MSE", "SMAPE"]
    date_flags = DateFlagsTransform(day_number_in_week=True, day_number_in_month=True)
    pipe = Pipeline(model=CatBoostMultiSegmentModel(), horizon=10, transforms=[date_flags])
    n_folds = 5
    pipe.backtest(ts=example_tsds, metrics=metrics, n_jobs=1, n_folds=n_folds)
    with open(file.name, "r") as in_file:
        lines = in_file.readlines()
        # remain lines only about backtest
        lines = [line for line in lines if "backtest" in line]
        assert len(lines) == len(metrics) * n_folds * len(example_tsds.segments)
        assert all([any([metric_str in line for metric_str in metrics_str]) for line in lines])
    tslogger.remove(idx)


def test_backtest_logging_no_tables(example_tsds: TSDataset):
    """Check working of logging inside backtest with `table=False`."""
    file = NamedTemporaryFile()
    _logger.add(file.name)
    idx = tslogger.add(ConsoleLogger(table=False))
    metrics = [MAE(), MSE(), SMAPE()]
    date_flags = DateFlagsTransform(day_number_in_week=True, day_number_in_month=True)
    pipe = Pipeline(model=CatBoostMultiSegmentModel(), horizon=10, transforms=[date_flags])
    n_folds = 5
    pipe.backtest(ts=example_tsds, metrics=metrics, n_jobs=1, n_folds=n_folds)
    with open(file.name, "r") as in_file:
        lines = in_file.readlines()
        # remain lines only about backtest
        lines = [line for line in lines if "backtest" in line]
        assert len(lines) == 0
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
        # filter out logs related to transforms
        lines = [line for line in lines if lags.__class__.__name__ not in line]
        assert len(lines) == 2
        assert "fit" in lines[0]
        assert "forecast" in lines[1]

    tslogger.remove(idx)
