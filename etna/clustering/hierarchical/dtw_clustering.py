from typing import TYPE_CHECKING

from etna.clustering.distances.dtw_distance import DTWDistance
from etna.clustering.hierarchical.base import HierarchicalClustering

if TYPE_CHECKING:
    from etna.datasets import TSDataset


class DTWClustering(HierarchicalClustering):
    """Hierarchical clustering with DTW distance.

    Examples
    --------
    >>> from etna.clustering import DTWClustering
    >>> from etna.datasets import TSDataset
    >>> from etna.datasets import generate_ar_df
    >>> ts = generate_ar_df(periods = 40, start_time = "2000-01-01", n_segments = 10)
    >>> ts = TSDataset(TSDataset.to_dataset(ts), freq="D")
    >>> model = DTWClustering()
    >>> model.build_distance_matrix(ts)
    >>> model.build_clustering_algo(n_clusters=3, linkage="average")
    >>> segment2cluster = model.fit_predict()
    >>> segment2cluster
    {'segment_0': 2,
     'segment_1': 1,
     'segment_2': 0,
     'segment_3': 1,
     'segment_4': 1,
     'segment_5': 0,
     'segment_6': 0,
     'segment_7': 1,
     'segment_8': 2,
     'segment_9': 2}
    """

    def __init__(self):
        """Create instance of DTWClustering."""
        super().__init__(distance=DTWDistance())

    def build_distance_matrix(self, ts: "TSDataset"):
        """
        Build distance matrix with DTW distance.

        Parameters
        ----------
        ts:
            TSDataset with series to build distance matrix
        """
        super().build_distance_matrix(ts=ts)


__all__ = ["DTWClustering"]
