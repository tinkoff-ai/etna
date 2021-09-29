from etna.clustering.distances.euclidean_distance import EuclideanDistance
from etna.clustering.hierarchical.base import HierarchicalClustering


class EuclideanClustering(HierarchicalClustering):
    """Hierarchical clustering with euclidean distance."""

    def build_distance_matrix(self, ts: "TSDataset"):
        """
        Build distance matrix with euclidean distance.

        Parameters
        ----------
        ts:
            TSDataset with series to build distance matrix
        """
        super().build_distance_matrix(ts=ts, distance=EuclideanDistance())


__all__ = ["EuclideanClustering"]
