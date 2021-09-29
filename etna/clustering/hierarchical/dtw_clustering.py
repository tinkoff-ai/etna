from etna.clustering.distances.dtw_distance import DTWDistance
from etna.clustering.hierarchical.base import HierarchicalClustering


class DTWClustering(HierarchicalClustering):
    """Hierarchical clustering with DTW distance."""

    def build_distance_matrix(self, ts: "TSDataset"):
        """
        Build distance matrix with DTW distance.

        Parameters
        ----------
        ts:
            TSDataset with series to build distance matrix
        """
        super().build_distance_matrix(ts=ts, distance=DTWDistance())


__all__ = ["DTWClustering"]
