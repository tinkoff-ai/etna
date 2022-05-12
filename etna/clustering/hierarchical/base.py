from enum import Enum
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import pandas as pd
from sklearn.cluster import AgglomerativeClustering

from etna.clustering.base import Clustering
from etna.clustering.distances.base import Distance
from etna.clustering.distances.distance_matrix import DistanceMatrix
from etna.datasets import TSDataset


class ClusteringLinkageMode(Enum):
    """Modes allowed for clustering distance computation."""

    ward = "ward"
    complete = "complete"
    average = "average"
    single = "single"


class HierarchicalClustering(Clustering):
    """Base class for hierarchical clustering."""

    def __init__(self, distance: Distance):
        """Init HierarchicalClustering."""
        super().__init__()
        self.n_clusters: Optional[int] = None
        self.linkage: Optional[str] = None
        self.clustering_algo: Optional[AgglomerativeClustering] = None
        self.distance_matrix: Optional[DistanceMatrix] = None
        self.clusters: Optional[List[int]] = None
        self.ts: Optional["TSDataset"] = None
        self.segment2cluster: Optional[Dict[str, int]] = None
        self.distance: Distance = distance
        self.centroids_df: Optional[pd.DataFrame] = None

    def build_distance_matrix(self, ts: "TSDataset"):
        """Compute distance matrix with given ts and distance.

        Parameters
        ----------
        ts:
            TSDataset with series to build distance matrix
        distance:
            instance if distance to compute matrix
        """
        self.ts = ts
        self.distance_matrix = DistanceMatrix(distance=self.distance)
        self.distance_matrix.fit(ts=ts)
        self.clusters = None
        self.segment2cluster = None
        self.centroids_df = None

    def build_clustering_algo(
        self,
        n_clusters: int = 30,
        linkage: Union[str, ClusteringLinkageMode] = ClusteringLinkageMode.average,
        **clustering_algo_params,
    ):
        """Build clustering algo (see :py:class:`sklearn.cluster.AgglomerativeClustering`) with given params.

        Parameters
        ----------
        n_clusters:
            number of clusters to build
        linkage:
            rule for distance computation for new clusters, allowed "ward", "single", "average", "maximum", "complete"

        Notes
        -----
        Note that it will reset previous results of clustering in case of reinit algo.
        """
        self.n_clusters = n_clusters
        self.linkage = ClusteringLinkageMode(linkage).name
        self.clustering_algo = AgglomerativeClustering(
            n_clusters=self.n_clusters, affinity="precomputed", linkage=self.linkage, **clustering_algo_params
        )
        self.clusters = None
        self.segment2cluster = None
        self.centroids_df = None

    def fit_predict(self) -> Dict[str, int]:
        """Fit clustering algorithm and predict clusters according to distance matrix build.

        Returns
        -------
        Dict[str, int]:
            dict in format {segment: cluster}
        """
        if self.clustering_algo is None:
            raise ValueError(
                "Clustering algorithm is not built! Build clustering algorithm using build_clustering_algo method before calling fit_predict!"
            )
        if self.distance_matrix is None:
            raise ValueError(
                "Distance matrix is not built! Build distance matrix using build_distance_matrix method before calling fit_predict!"
            )
        self.clusters = self.clustering_algo.fit_predict(X=self.distance_matrix.matrix)
        if self.clusters is None:
            raise ValueError("Something went wrong during predicting the clusters!")
        self.segment2cluster = {
            self.distance_matrix.idx2segment[i]: self.clusters[i] for i in range(len(self.clusters))
        }
        return self.segment2cluster

    def _get_series_in_cluster(self, cluster: int) -> TSDataset:
        """Get series in cluster."""
        if self.ts is None or self.segment2cluster is None:
            raise ValueError(
                "HierarchicalClustering is not fitted! Fit the HierarchicalClustering before calling get_centroids!"
            )
        segments_in_cluster = [segment for segment in self.ts.segments if self.segment2cluster[segment] == cluster]
        cluster_ts = TSDataset(df=self.ts[:, segments_in_cluster, "target"], freq=self.ts.freq)
        return cluster_ts

    def _get_centroid_of_cluster(self, cluster: int, **averaging_kwargs) -> pd.DataFrame:
        """Get centroid of cluster."""
        cluster_ts = self._get_series_in_cluster(cluster)
        centroid = self.distance.get_average(ts=cluster_ts, **averaging_kwargs)
        centroid["segment"] = cluster
        return centroid

    def get_centroids(self, **averaging_kwargs) -> pd.DataFrame:
        """Get centroids of clusters.

        Returns
        -------
        pd.DataFrame:
            dataframe with centroids
        """
        if self.clusters is None:
            raise ValueError(
                "HierarchicalClustering is not fitted! Fit the HierarchicalClustering before calling get_centroids!"
            )
        centroids = []
        clusters = set(self.clusters)
        for cluster in clusters:
            centroid = self._get_centroid_of_cluster(cluster=cluster, **averaging_kwargs)
            centroids.append(centroid)
        self.centroids_df = pd.concat(centroids, ignore_index=True)
        self.centroids_df = TSDataset.to_dataset(self.centroids_df)

        # modify dataframe columns
        columns_frame = self.centroids_df.columns.to_frame()
        columns_frame["segment"] = columns_frame["segment"].astype(int)
        self.centroids_df.columns = pd.MultiIndex.from_frame(columns_frame)
        self.centroids_df.columns.set_names("cluster", level=0, inplace=True)
        return self.centroids_df


__all__ = ["HierarchicalClustering", "ClusteringLinkageMode"]
