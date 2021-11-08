from typing import Union

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeRegressor

from etna.libs.tsfresh import calculate_relevance_table

TreeBasedRegressor = Union[
    DecisionTreeRegressor,
    ExtraTreeRegressor,
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    CatBoostRegressor,
]


def get_statistics_relevance_table(df: pd.DataFrame, df_exog: pd.DataFrame) -> pd.DataFrame:
    """Calculate relevance table with p-values from tsfresh.

    Parameters
    ----------
    df:
        dataframe with timeseries
    df_exog:
        dataframe with exogenous data

    Returns
    -------
    dataframe with p-values.
    """
    regressors = sorted(df_exog.columns.get_level_values("feature").unique())
    segments = sorted(df.columns.get_level_values("segment").unique())
    result = np.empty((len(segments), len(regressors)))
    for k, seg in enumerate(segments):
        first_valid_idx = df.loc[:, seg].first_valid_index()
        df_now = df.loc[first_valid_idx:, seg]["target"]
        df_exog_now = df_exog.loc[first_valid_idx:, seg]
        relevance = calculate_relevance_table(df_exog_now[: len(df_now)], df_now)[["feature", "p_value"]].values
        result[k] = np.array(sorted(relevance, key=lambda x: x[0]))[:, 1]
    relevance_table = pd.DataFrame(result)
    relevance_table.index = segments
    relevance_table.columns = regressors
    return relevance_table


def get_model_relevance_table(df: pd.DataFrame, df_exog: pd.DataFrame, model: TreeBasedRegressor) -> pd.DataFrame:
    """Calculate relevance table with feature importance from model.

    Parameters
    ----------
    df:
        dataframe with timeseries
    df_exog:
        dataframe with exogenous data
    model:
        model to obtain feature importance, should have feature_importances_ property

    Returns
    -------
    dataframe with feature importance values.
    """
    regressors = sorted(df_exog.columns.get_level_values("feature").unique())
    segments = sorted(df.columns.get_level_values("segment").unique())
    result = np.empty((len(segments), len(regressors)))
    for k, seg in enumerate(segments):
        first_valid_idx = df.loc[:, seg].first_valid_index()
        df_now = df.loc[first_valid_idx:, seg]["target"]
        df_exog_now = df_exog.loc[first_valid_idx : df_now.index.max(), seg][regressors]
        model.fit(df_exog_now, df_now)
        result[k] = model.feature_importances_
    relevance_table = pd.DataFrame(result)
    relevance_table.index = segments
    relevance_table.columns = regressors
    return relevance_table
