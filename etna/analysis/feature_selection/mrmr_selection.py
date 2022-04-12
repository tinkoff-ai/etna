from enum import Enum
from typing import List

import numpy as np
import pandas as pd


class AggregationMode(str, Enum):
    """Enum for different aggregation modes."""

    mean = "mean"
    max = "max"
    min = "min"
    median = "median"


AGGREGATION_FN = {
    AggregationMode.mean: np.mean,
    AggregationMode.max: np.max,
    AggregationMode.min: np.min,
    AggregationMode.median: np.median,
}


def mrmr(
    relevance_table: pd.DataFrame,
    regressors: pd.DataFrame,
    top_k: int,
    relevance_aggregation_mode: str = AggregationMode.mean,
    redundancy_aggregation_mode: str = AggregationMode.mean,
    atol: float = 1e-10,
) -> List[str]:
    """
    Maximum Relevance and Minimum Redundancy feature selection method.

    Here relevance for each regressor is calculated as the per-segment aggregation of the relevance
    values in relevance_table. The redundancy term for the regressor is calculated as a mean absolute correlation
    between this regressor and other ones. The correlation between the two regressors is an aggregated pairwise
    correlation for the regressors values in each segment.

    Parameters
    ----------
    relevance_table:
        dataframe of shape n_segment x n_exog_series with relevance table, where ``relevance_table[i][j]``
        contains relevance of j-th ``df_exog`` series to i-th df series
    regressors:
        dataframe with regressors in etna format
    top_k:
        num of regressors to select; if there are not enough regressors, then all will be selected
    relevance_aggregation_mode:
        the method for relevance values per-segment aggregation
    redundancy_aggregation_mode:
        the method for redundancy values per-segment aggregation
    atol:
        the absolute tolerance to compare the float values

    Returns
    -------
    selected_features: List[str]
        list of ``top_k`` selected regressors, sorted by their importance
    """
    relevance_aggregation_fn = AGGREGATION_FN[AggregationMode(relevance_aggregation_mode)]
    redundancy_aggregation_fn = AGGREGATION_FN[AggregationMode(redundancy_aggregation_mode)]

    relevance = relevance_table.apply(relevance_aggregation_fn).fillna(0)

    all_features = relevance.index.to_list()
    selected_features: List[str] = []
    not_selected_features = all_features.copy()

    redundancy_table = pd.DataFrame(np.inf, index=all_features, columns=all_features)
    top_k = min(top_k, len(all_features))

    for i in range(top_k):
        score_numerator = relevance.loc[not_selected_features]
        score_denominator = pd.Series(1, index=not_selected_features)
        if i > 0:
            last_selected_feature = selected_features[-1]
            not_selected_regressors = regressors.loc[pd.IndexSlice[:], pd.IndexSlice[:, not_selected_features]]
            last_selected_regressor = regressors.loc[pd.IndexSlice[:], pd.IndexSlice[:, last_selected_feature]]

            redundancy_table.loc[not_selected_features, last_selected_feature] = (
                not_selected_regressors.apply(lambda col: last_selected_regressor.corrwith(col))
                .abs()
                .groupby("feature")
                .apply(redundancy_aggregation_fn)
                .T.groupby("feature")
                .apply(redundancy_aggregation_fn)
                .clip(atol)
                .fillna(np.inf)
                .loc[not_selected_features]
                .values.squeeze()
            )

            score_denominator = redundancy_table.loc[not_selected_features, selected_features].mean(axis=1)
            score_denominator[np.isclose(score_denominator, 1, atol=atol)] = np.inf
        score = score_numerator / score_denominator
        best_feature = score.index[score.argmax()]
        selected_features.append(best_feature)
        not_selected_features.remove(best_feature)

    return selected_features
