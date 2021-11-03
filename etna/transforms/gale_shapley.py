from collections import defaultdict
from typing import List, Dict

from etna.analysis import RelevanceTable, StatisticsRelevanceTable
from etna.transforms import Transform
import warnings
import pandas as pd
from math import ceil


class GaleShapleyFeatureSelectionTransform(Transform):
    def __init__(self, relevance_table: RelevanceTable, top_k: int):
        self.relevance_table = relevance_table
        self.top_k = top_k

    @staticmethod
    def _get_regressors(df: pd.DataFrame) -> List[str]:
        """Get list of regressors in the dataframe."""
        result = set()
        for column in df.columns.get_level_values("feature"):
            if column.startswith("regressor_"):
                result.add(column)
        return sorted(list(result))

    def _compute_relevance_table(self, df, regressors: List[str]) -> pd.DataFrame:
        targets_df = df.loc[:, pd.IndexSlice[:, "target"]]
        regressors_df = df.loc[:, pd.IndexSlice[:, regressors]]
        table = self.relevance_table(df=targets_df, df_exog=regressors_df)
        return table

    @staticmethod
    def _get_ranked_list(table: pd.DataFrame, ascending: bool) -> Dict[str, List[str]]:
        ranked_regressors = {
            key: list(table.loc[key].sort_values(ascending=ascending).index) for key in table.index
        }
        return ranked_regressors

    @staticmethod
    def _compute_gale_shapley_steps_number(top_k: int, n_segments: int, n_regressors: int) -> int:
        if n_regressors < top_k:
            warnings.warn(
                f"Given top_k={top_k} is bigger than n_regressors={n_regressors}. "
                f"Transform will not filter regressors."
            )
            return 1
        if top_k < n_segments:
            warnings.warn(
                f"Given top_k={top_k} is less than n_segments. Algo will filter data without Gale-Shapley run."
            )
            return 1
        return ceil(top_k / n_segments)

    def _gale_shapley_iteration(self, segment_regressors_ranking: Dict[str, List[str]]) -> Dict[str, str]:
        """Build matching for all the regressors.

        Parameters
        ----------
        segment_regressors_ranking:
            dict of relevance segment x sorted regressors

        Returns
        -------
        matching dict: Dict[str, str]
            dict of segment x regressor
        """
        pass

    def _update_ranking_list(self, segment_regressors_ranking: Dict[str, List[str]], regressors_to_drop: List[str]):
        pass

    def _process_last_step(self, ):
        pass

    def fit_transform(self, df: pd.DataFrame):
        regressors = self._get_regressors(df=df)
        relevance_table = self._compute_relevance_table(df=df, regressors=regressors)
        segment_regressors_ranking = self._get_ranked_list(
            table=relevance_table,
            ascending=not self.relevance_table.greater_is_better
        )
        regressor_segments_ranking = self._get_ranked_list(
            table=relevance_table.T,
            ascending=not self.relevance_table.greater_is_better
        )
        gale_shapley_steps_number = self._compute_gale_shapley_steps_number(
            top_k=self.top_k,
            n_segments=len(segment_regressors_ranking),
            n_regressors=len(regressor_segments_ranking),
        )