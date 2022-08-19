from typing import Dict
from typing import List

import pandas as pd
from ruptures.base import BaseEstimator

from etna.datasets import TSDataset


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
        params for ``change_point_model`` predict method

    Returns
    -------
    Dict[str, List[pd.Timestamp]]
        dictionary with list of trend change points for each segment
    """
    from etna.transforms.decomposition.base_change_points import RupturesChangePointsModel

    result: Dict[str, List[pd.Timestamp]] = {}
    df = ts.to_pandas()
    ruptures = RupturesChangePointsModel(change_point_model, **model_predict_params)
    for segment in ts.segments:
        df_segment = df[segment]
        result[segment] = ruptures.get_change_points(df=df_segment, in_column=in_column)
    return result
