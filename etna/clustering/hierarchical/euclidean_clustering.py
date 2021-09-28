import pandas as pd

from etna.clustering.distances.euclidean_distance import EuclideanDistance
from etna.clustering.hierarchical.base import HierarchicalClustering


class EuclideanClustering(HierarchicalClustering):
    """Hierarchical clustering with euclidean distance."""

    def build_distance_matrix(self, df: pd.DataFrame):
        """
        Build distance matrix with euclidean distance.

        Parameters
        ----------
        df: pd.DataFrame
            dataframe with series to build distance matrix
        """
        super().build_distance_matrix(df=df, distance=EuclideanDistance())


__all__ = ["EuclideanClustering"]
