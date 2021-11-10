from abc import ABC
from abc import abstractmethod

import pandas as pd

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

    @abstractmethod
    def __call__(self, df: pd.DataFrame, df_exog: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Compute relevance table.
        For each series in df compute relevance of corresponding series in df_exog.

        Parameters
        ----------
        df:
            dataframe with series that will be used as target
        df_exog:
            dataframe with series to compute relevance for df

        Returns
        -------
        relevance table: pd.DataFrame
            dataframe of shape n_segment x n_exog_series, relevance_table[i][j] contains relevance of j-th df_exog series to i-th df series
        """
        pass


class StatisticsRelevanceTable(RelevanceTable):
    """StatisticsRelevanceTable builds feature relevance table with tsfresh statistics."""

    def __init__(self):
        super().__init__(greater_is_better=False)

    def __call__(self, df: pd.DataFrame, df_exog: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Compute feature relevance table with etna.analysis.get_statistics_relevance_table method."""
        table = get_statistics_relevance_table(df=df, df_exog=df_exog)
        return table


class ModelRelevanceTable(RelevanceTable):
    """ModelRelevanceTable builds feature relevance table using feature relevance values obtained from model."""

    def __init__(self):
        super().__init__(greater_is_better=True)

    def __call__(self, df: pd.DataFrame, df_exog: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Compute feature relevance table with etna.analysis.get_model_relevance_table method."""
        table = get_model_relevance_table(df=df, df_exog=df_exog, **kwargs)
        return table
