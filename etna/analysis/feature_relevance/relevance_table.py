import warnings
from typing import List
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


def _prepare_df(df: pd.DataFrame, df_exog: pd.DataFrame, segment: str, regressors: List[str]):
    """Drop nan values from dataframes for the segment."""
    first_valid_idx = df.loc[:, segment].first_valid_index()
    df_exog_seg = df_exog.loc[first_valid_idx:, segment].dropna()[regressors]
    df_seg = df.loc[first_valid_idx:, segment].dropna()["target"]
    common_index = df_seg.index.intersection(df_exog_seg.index)
    if len(common_index) < len(df.loc[first_valid_idx:, segment]):
        warnings.warn("Exogenous or target data contains None! It will be dropped for calculating relevance.")
    return df_seg.loc[common_index], df_exog_seg.loc[common_index]


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
    for k, seg in enumerate(segments):
        df_seg, df_exog_seg = _prepare_df(df=df, df_exog=df_exog, segment=seg, regressors=regressors)
        cat_cols = df_exog_seg.dtypes[df_exog_seg.dtypes == "category"].index
        for cat_col in cat_cols:
            try:
                df_exog_seg[cat_col] = df_exog_seg[cat_col].astype(float)
            except ValueError:
                raise ValueError(f"{cat_col} column cannot be cast to float type! Please, use encoders.")
            warnings.warn(
                "Exogenous data contains columns with category type! It will be converted to float. If this is not desired behavior, use encoders."
            )

        relevance = calculate_relevance_table(X=df_exog_seg, y=df_seg)[["feature", "p_value"]].values
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
        df_seg, df_exog_seg = _prepare_df(df=df, df_exog=df_exog, segment=seg, regressors=regressors)
        model.fit(X=df_exog_seg, y=df_seg)
        result[k] = model.feature_importances_
    relevance_table = pd.DataFrame(result)
    relevance_table.index = segments
    relevance_table.columns = regressors
    return relevance_table
