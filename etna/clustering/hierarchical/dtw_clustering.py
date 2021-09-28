import pandas as pd

from etna.clustering.distances.dtw_distance import DTWDistance
from etna.clustering.hierarchical.base import HierarchicalClustering


class DTWClustering(HierarchicalClustering):
    """Hierarchical clustering with DTW distance."""

    def build_distance_matrix(self, df: pd.DataFrame):
        """
        Build distance matrix with DTW distance.

        Parameters
        ----------
        df: pd.DataFrame
            dataframe with series to build distance matrix
        """
        super().build_distance_matrix(df=df, distance=DTWDistance())


__all__ = ["DTWClustering"]
