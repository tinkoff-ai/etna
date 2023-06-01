from enum import Enum
from math import floor
from typing import Literal
from typing import Optional
from typing import Union

from etna.datasets import TSDataset
from etna.pipeline import Pipeline


class MethodsWithFolds(str, Enum):
    """Enum for methods that use `n_folds` argument."""

    forecast = "forecast"
    backtest = "backtest"

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not a valid method name. Only {', '.join([repr(m.value) for m in cls])} are allowed"
        )


def _estimate_n_folds(num_points: int, horizon: int, stride: int, context_size: int) -> int:
    """Estimate number of folds."""
    if num_points < horizon + context_size:
        raise ValueError("Not enough data points!")

    res = (num_points - horizon + stride - context_size) / stride
    return floor(res)


def _max_n_folds_forecast(pipeline: Pipeline, context_size: int, ts: Optional[TSDataset] = None) -> int:
    """Estimate max n_folds for forecast method."""
    if ts is None:
        if pipeline.ts is None:
            raise ValueError(
                "There is no ts for forecast method! Pass ts into function or make sure that pipeline is fitted."
            )

        else:
            ts = pipeline.ts

    num_points = len(ts.index)
    horizon = pipeline.horizon

    return _estimate_n_folds(num_points=num_points, horizon=horizon, stride=horizon, context_size=context_size)


def _max_n_folds_backtest(pipeline: Pipeline, context_size: int, ts: TSDataset, **method_kwargs) -> int:
    """Estimate max n_folds for backtest method."""
    # process backtest with intervals case
    backtest_with_intervals = "forecast_params" in method_kwargs and method_kwargs["forecast_params"].get(
        "prediction_interval", False
    )

    if backtest_with_intervals:
        raise NotImplementedError("Number of folds estimation for backtest with intervals is not implemented!")

    num_points = len(ts.index)

    horizon = pipeline.horizon
    stride = method_kwargs.get("stride", horizon)

    return _estimate_n_folds(num_points=num_points, horizon=horizon, stride=stride, context_size=context_size)


def estimate_max_n_folds(
    pipeline: Pipeline,
    method_name: Union[Literal["forecast"], Literal["backtest"]],
    context_size: int,
    ts: Optional[TSDataset] = None,
    **method_kwargs,
) -> int:
    """Estimate number of folds using provided data and pipeline configuration.

    Parameters
    ----------
    pipeline:
        pipeline for which to estimate number of folds.
    method_name:
       method name for which to estimate number of folds.
    context_size:
       minimum number of points for pipeline to be estimated.
    ts:
       dataset which will be used for estimation.
    method_kwargs:
       additional arguments for methods that impact number of folds.

    Returns
    -------
    :
        Number of folds.
    """
    if context_size < 1:
        raise ValueError("Pipeline `context_size` parameter must be positive integer!")

    if ts is None and method_name != MethodsWithFolds.forecast:
        raise ValueError("Parameter `ts` is required when estimating for backtest method")

    if ts is not None and len(ts.index) == 0:
        raise ValueError("Empty ts is passed!")

    method = MethodsWithFolds(method_name)

    if method == MethodsWithFolds.forecast:
        n_folds = _max_n_folds_forecast(pipeline=pipeline, context_size=context_size, ts=ts)

    else:
        n_folds = _max_n_folds_backtest(pipeline=pipeline, context_size=context_size, ts=ts, **method_kwargs)  # type: ignore

    return n_folds
