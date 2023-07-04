from enum import Enum
from typing import Dict
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
from ruptures.base import BaseEstimator
from ruptures.costs import CostLinear

from etna.datasets import TSDataset


class OptimizationMode(str, Enum):
    """Enum for different optimization modes."""

    pen = "pen"
    epsilon = "epsilon"

    @classmethod
    def _missing_(cls, value):
        raise NotImplementedError(
            f"{value} is not a valid {cls.__name__}. Only {', '.join([repr(m.value) for m in cls])} modes allowed"
        )


def _get_n_bkps(series: pd.Series, change_point_model: BaseEstimator, **model_predict_params) -> int:
    """Get number of change points, detected with given params.

    Parameters
    ----------
    series:
        series to detect change points
    change_point_model:
        model to get trend change points

    Returns
    -------
    :
        number of change points
    """
    signal = series.to_numpy()
    if isinstance(change_point_model.cost, CostLinear):
        signal = signal.reshape((-1, 1))

    change_point_model.fit(signal=signal)

    change_points_indices = change_point_model.predict(**model_predict_params)[:-1]
    return len(change_points_indices)


def _get_next_value(
    now_value: float, lower_bound: float, upper_bound: float, need_greater: bool
) -> Tuple[float, float, float]:
    """Give next value according to binary search.

    Parameters
    ----------
    now_value:
        current value
    lower_bound:
        lower bound for search
    upper_bound:
        upper bound for search
    need_greater:
        True if we need greater value for n_bkps than previous time

    Returns
    -------
    :
        next value and its bounds
    """
    if need_greater:
        return np.mean([now_value, lower_bound]), lower_bound, now_value
    else:
        return np.mean([now_value, upper_bound]), now_value, upper_bound


def bin_search(
    series: pd.Series,
    change_point_model: BaseEstimator,
    n_bkps: int,
    opt_param: str,
    max_value: float,
    max_iters: int = 200,
) -> float:
    """Run binary search for optimal regularizations.

    Parameters
    ----------
    series:
        series for search
    change_point_model:
        model to get trend change points
    n_bkps:
        target numbers of changepoints
    opt_param:
        parameter for optimization
    max_value:
        maximum possible value, the upper bound for search
    max_iters:
        maximum iterations; in case if the required number of points is unattainable, values will be selected after max_iters iterations

    Returns
    -------
    :
        regularization parameters value

    Raises
    ______
    ValueError:
        If max_value is too low for needed n_bkps
    ValueError:
        If n_bkps is too high for this series
    """
    zero_param = _get_n_bkps(series, change_point_model, **{opt_param: 0})
    max_param = _get_n_bkps(series, change_point_model, **{opt_param: max_value})
    if zero_param < n_bkps:
        raise ValueError("Impossible number of changepoints. Please, decrease n_bkps value.")
    if n_bkps < max_param:
        raise ValueError("Impossible number of changepoints. Please, increase max_value or increase n_bkps value.")

    lower_bound, upper_bound = 0.0, max_value
    now_value = np.mean([lower_bound, upper_bound])
    now_n_bkps = _get_n_bkps(series, change_point_model, **{opt_param: now_value})
    iters = 0

    while now_n_bkps != n_bkps and iters < max_iters:
        need_greater = now_n_bkps < n_bkps
        now_value, lower_bound, upper_bound = _get_next_value(now_value, lower_bound, upper_bound, need_greater)
        now_n_bkps = _get_n_bkps(series, change_point_model, **{opt_param: now_value})
        iters += 1
    return now_value


def get_ruptures_regularization(
    ts: TSDataset,
    in_column: str,
    change_point_model: BaseEstimator,
    n_bkps: Union[Dict[str, int], int],
    mode: OptimizationMode,
    max_value: float = 10000,
    max_iters: int = 200,
) -> Dict[str, Dict[str, float]]:
    """Get regularization parameter values for given number of changepoints.

    It is assumed that as the regularization being selected increases, the number of change points decreases.

    Parameters
    ----------
    ts:
        Dataset with timeseries data
    in_column:
        name of processed column
    change_point_model:
        model to get trend change points
    n_bkps:
        target numbers of changepoints
    mode:
        optimization mode
    max_value:
        maximum possible value, the upper bound for search
    max_iters:
        maximum iterations; in case if the required number of points is unattainable, values will be selected after max_iters iterations

    Returns
    -------
    :
        regularization parameters values in dictionary format {segment: {mode: value}}.

    Raises
    ______
    ValueError:
        If max_value is too low for needed n_bkps
    ValueError:
        If n_bkps is too high for this series
    """
    mode = OptimizationMode(mode)
    df = ts.to_pandas()
    segments = df.columns.get_level_values(0).unique()

    if isinstance(n_bkps, int):
        n_bkps = dict(zip(segments, [n_bkps] * len(segments)))

    regulatization = {}
    for segment in segments:
        series = ts[:, segment, in_column]
        regulatization[segment] = {
            mode.value: bin_search(series, change_point_model, n_bkps[segment], mode, max_value, max_iters)
        }
    return regulatization
