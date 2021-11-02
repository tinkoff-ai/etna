import pandas as pd
from typing import Callable
from etna.analysis.feature_relevance.relevance_functions import get_statistics_relevance_table
from abc import ABC, abstractmethod


class RelevanceTable(ABC):
    def __init__(self, greater_is_better: bool):
        self.greater_is_better = greater_is_better

    @abstractmethod
    def __call__(self, *args, **kwargs) -> pd.DataFrame:
        pass


class StatisticsRelevanceTable(RelevanceTable):
    def __call__(self, df: pd.DataFrame, df_exog: pd.DataFrame) -> pd.DataFrame:
        table = get_statistics_relevance_table(df=df, df_exog=df_exog)
        return table
