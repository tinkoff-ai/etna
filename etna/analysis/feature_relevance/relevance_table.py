import warnings
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
    pd.DataFrame
        dataframe with p-values.
    """
    regressors = sorted(df_exog.columns.get_level_values("feature").unique())
    segments = sorted(df.columns.get_level_values("segment").unique())
    result = np.empty((len(segments), len(regressors)))
    none_warning_raised = False
    category_warning_raised = False
    for k, seg in enumerate(segments):
        first_valid_idx = df.loc[:, seg].first_valid_index()
        df_now = df.loc[first_valid_idx:, seg]["target"]
        df_exog_now = df_exog.loc[first_valid_idx:, seg][: len(df_now)]
        cat_cols = df_exog_now.dtypes[df_exog_now.dtypes == "category"].index
        for cat_col in cat_cols:
            try:
                df_exog_now[cat_col] = df_exog_now[cat_col].astype(float)
            except ValueError:
                raise ValueError(f"{cat_col} column cannot be cast to float type! Please, use encoders.")
        if len(cat_cols) > 0 and not category_warning_raised:
            category_warning_raised = True
            warnings.warn(
                "Exogenous data contains columns with category type! It will be converted to float. If this is not desired behavior, use encoders."
            )
        df_exog_now["target"] = df_now
        df_exog_now = df_exog_now.dropna()
        if len(df_exog_now) != len(df_now) and not none_warning_raised:
            none_warning_raised = True
            warnings.warn("Exogenous or target data contains None! It will be dropped for calculating relevance.")
        relevance = calculate_relevance_table(df_exog_now.drop(columns=["target"]), df_exog_now["target"])[
            ["feature", "p_value"]
        ].values
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
        model to obtain feature importance, should have ``feature_importances_`` property

    Returns
    -------
    pd.DataFrame
        dataframe with feature importance values.
    """
    regressors = sorted(df_exog.columns.get_level_values("feature").unique())
    segments = sorted(df.columns.get_level_values("segment").unique())
    result = np.empty((len(segments), len(regressors)))
    for k, seg in enumerate(segments):
        df_exog_seg = df_exog.loc[:, seg].dropna()[regressors]
        df_seg = df.loc[:, seg].dropna()["target"]
        common_index = df_seg.index.intersection(df_exog_seg.index)
        model.fit(df_exog_seg.loc[common_index], df_seg.loc[common_index])
        result[k] = model.feature_importances_
    relevance_table = pd.DataFrame(result)
    relevance_table.index = segments
    relevance_table.columns = regressors
    return relevance_table
