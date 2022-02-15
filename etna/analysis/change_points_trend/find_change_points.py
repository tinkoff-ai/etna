from typing import Dict
from typing import List

import numpy as np
import pandas as pd
from ruptures.base import BaseEstimator
from ruptures.costs import CostLinear

from etna.datasets import TSDataset


def _prepare_signal(series: pd.Series, model: BaseEstimator) -> np.ndarray:
    """Prepare series for change point model."""
    signal = series.to_numpy()
    if isinstance(model.cost, CostLinear):
        signal = signal.reshape((-1, 1))
    return signal


def _find_change_points_segment(
    series: pd.Series, change_point_model: BaseEstimator, **model_predict_params
) -> List[pd.Timestamp]:
    """Find trend change points within one segment."""
    signal = _prepare_signal(series=series, model=change_point_model)
    timestamp = series.index
    change_point_model.fit(signal=signal)
    # last point in change points is the first index after the series
    change_points_indices = change_point_model.predict(**model_predict_params)[:-1]
    change_points = [timestamp[idx] for idx in change_points_indices]
    return change_points


def find_change_points(
    ts: TSDataset, in_column: str, change_point_model: BaseEstimator, **model_predict_params
) -> Dict[str, List[pd.Timestamp]]:
    """Find trend change points using ruptures models.

    Parameters
    ----------
    ts:
        dataset to work with
    in_column:
        name of column to work with
    change_point_model:
        ruptures model to get trend change points
    model_predict_params:
        params for change_point_model predict method

    Returns
    -------
    result:
        dictionary with list of trend change points for each segment
    """
    result: Dict[str, List[pd.Timestamp]] = {}
    df = ts.to_pandas()
    for segment in ts.segments:
        df_segment = df[segment]
        raw_series = df_segment[in_column]
        series = raw_series.loc[raw_series.first_valid_index() : raw_series.last_valid_index()]
        result[segment] = _find_change_points_segment(
            series=series, change_point_model=change_point_model, **model_predict_params
        )
    return result
