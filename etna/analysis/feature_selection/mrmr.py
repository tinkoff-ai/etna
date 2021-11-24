import warnings
from multiprocessing import cpu_count
from typing import List

import numpy as np
import pandas as pd
from joblib import Parallel
from joblib import delayed
from sklearn.feature_selection import f_classif as sklearn_f_classif

warnings.filterwarnings("ignore")

FLOOR = 0.00001


def parallel_df(func, df, series):
    """Parallelize functions."""
    n_jobs = min(cpu_count(), len(df.columns))
    col_chunks = np.array_split(range(len(df.columns)), n_jobs)
    lst = Parallel(n_jobs=n_jobs)(delayed(func)(df.iloc[:, col_chunk], series) for col_chunk in col_chunks)
    return pd.concat(lst)


def mrmr(x: pd.DataFrame, y: np.ndarray, k: int) -> List[str]:
    """
    Maximum Relevance and Minimum Redundancy feature selection method.

    Parameters:
    ----------
    X:
        dataframe of shape n_segment x n_exog_series with relevance table, where relevance_table[i][j] contains relevance
        of j-th df_exog series to i-th df series
    y:
       class(cluster) labels of the segments
    K:
        num of regressors to select; if there are not enough regressors, then all will be selected

    Returns:
    -------
    selected_features: List[str]
        list of `top_k` selected regressors, sorted by their importance
    """
    x = x.dropna(axis=1)
    relevance_table = x.apply(lambda col: sklearn_f_classif(col[~col.isna()].to_frame(), y[~col.isna()])[0][0])
    relevance_table = relevance_table[relevance_table > 0]

    all_features = relevance_table.index.to_list()
    selected_features: List[str] = []
    not_selected_features = all_features.copy()

    redundancy_table = pd.DataFrame(FLOOR, index=all_features, columns=all_features)
    k = min(k, len(all_features))

    for i in range(k):
        score_numerator = relevance_table.loc[not_selected_features]
        score_denominator = pd.Series(1, index=not_selected_features)
        if i > 0:
            last_selected_feature = selected_features[-1]
            redundancy_table.loc[not_selected_features, last_selected_feature] = (
                x[not_selected_features].corrwith(x[last_selected_feature]).abs().clip(FLOOR).fillna(FLOOR)
            )
            score_denominator = (
                redundancy_table.loc[not_selected_features, selected_features]
                .mean(axis=1)
                .round(5)
                .replace(1.0, float("Inf"))
            )
        score = score_numerator / score_denominator
        best_feature = score.index[score.argmax()]
        selected_features.append(best_feature)
        not_selected_features.remove(best_feature)

    return selected_features
