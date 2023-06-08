from enum import Enum
from math import floor
from typing import Any
from typing import Dict
from typing import Literal
from typing import Optional
from typing import Set
from typing import Union

from etna.datasets import TSDataset
from etna.pipeline import Pipeline


def remove_params(params: Dict[str, Any], to_remove: Set[str]) -> Dict[str, Any]:
    """Select `forecast` arguments from params."""
    return {k: v for k, v in params.items() if k not in to_remove}


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

    This function helps to estimate maximum number of folds that can be used when performing
    forecast with intervals or pipeline backtest. Number of folds estimated using the following formula:

    .. math::
        max\\_n\\_folds = \\left\\lfloor\\frac{num\\_points - horizon + stride - context\\_size}{stride}\\right\\rfloor,

    where :math:`num\\_points` is number of points in the dataset,
    :math:`horizon` is length of forecasting horizon,
    :math:`stride` is number of points between folds,
    :math:`context\\_size` is pipeline context size.


    Parameters
    ----------
    pipeline:
        Pipeline for which to estimate number of folds.
    method_name:
        Method name for which to estimate number of folds.
    context_size:
        Minimum number of points for pipeline to be estimated.
    ts:
        Dataset which will be used for estimation.
    method_kwargs:
        Additional arguments for methods that impact number of folds.

    Returns
    -------
    :
        Number of folds.
    """
    if context_size < 1:
        raise ValueError("Pipeline `context_size` parameter must be positive integer!")

    if ts is None and method_name != MethodsWithFolds.forecast:
        raise ValueError("Parameter `ts` is required when estimating for backtest method")

    method = MethodsWithFolds(method_name)

    if method == MethodsWithFolds.forecast:
        n_folds = _max_n_folds_forecast(pipeline=pipeline, context_size=context_size, ts=ts)

    else:
        # ts always not None for backtest case
        n_folds = _max_n_folds_backtest(pipeline=pipeline, context_size=context_size, ts=ts, **method_kwargs)  # type: ignore

    return n_folds
