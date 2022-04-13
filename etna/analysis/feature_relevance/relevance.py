from abc import ABC
from abc import abstractmethod

import pandas as pd
import scipy.stats

from etna.analysis.feature_relevance.relevance_table import get_model_relevance_table
from etna.analysis.feature_relevance.relevance_table import get_statistics_relevance_table
from etna.core.mixins import BaseMixin


class RelevanceTable(ABC, BaseMixin):
    """Abstract class for relevance table computation."""

    def __init__(self, greater_is_better: bool):
        """Init RelevanceTable.

        Parameters
        ----------
        greater_is_better:
            bool flag, if True the biggest value in relevance table corresponds to the most important exog feature
        """
        self.greater_is_better = greater_is_better

    def _get_ranks(self, table: pd.DataFrame) -> pd.DataFrame:
        """Compute rank relevance table from relevance table."""
        if self.greater_is_better:
            table *= -1
        rank_table = pd.DataFrame(scipy.stats.rankdata(table, axis=1), columns=table.columns, index=table.index)
        return rank_table.astype(int)

    @abstractmethod
    def __call__(self, df: pd.DataFrame, df_exog: pd.DataFrame, return_ranks: bool = False, **kwargs) -> pd.DataFrame:
        """Compute relevance table.

        For each series in ``df`` compute relevance of corresponding series in ``df_exog``.

        Parameters
        ----------
        df:
            dataframe with series that will be used as target
        df_exog:
            dataframe with series to compute relevance for df
        return_ranks:
            if False return relevance values else return ranks of relevance values

        Returns
        -------
        relevance table:
            dataframe of shape n_segment x n_exog_series,
            ``relevance_table[i][j]`` contains relevance of j-th df_exog series to i-th df series
        """
        pass


class StatisticsRelevanceTable(RelevanceTable):
    """StatisticsRelevanceTable builds feature relevance table with tsfresh statistics."""

    def __init__(self):
        super().__init__(greater_is_better=False)

    def __call__(self, df: pd.DataFrame, df_exog: pd.DataFrame, return_ranks: bool = False, **kwargs) -> pd.DataFrame:
        """Compute feature relevance table with :py:func:`~etna.analysis.get_statistics_relevance_table` method."""
        table = get_statistics_relevance_table(df=df, df_exog=df_exog)
        if return_ranks:
            return self._get_ranks(table)
        return table


class ModelRelevanceTable(RelevanceTable):
    """ModelRelevanceTable builds feature relevance table using feature relevance values obtained from model."""

    def __init__(self):
        super().__init__(greater_is_better=True)

    def __call__(self, df: pd.DataFrame, df_exog: pd.DataFrame, return_ranks: bool = False, **kwargs) -> pd.DataFrame:
        """Compute feature relevance table with :py:func:`~etna.analysis.get_model_relevance_table` method."""
        table = get_model_relevance_table(df=df, df_exog=df_exog, **kwargs)
        if return_ranks:
            return self._get_ranks(table)
        return table
