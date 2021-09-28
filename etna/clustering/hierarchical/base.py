from typing import Dict
from typing import List
from typing import Optional

import pandas as pd
from sklearn.cluster import AgglomerativeClustering

from etna.clustering.base import Clustering
from etna.clustering.distances.base import Distance
from etna.clustering.distances.distance_matrix import DistanceMatrix


class ClusteringLinkageMode:
    """Modes allowed for clustering distance computation."""

    ward = "ward"
    complete = "complete"
    average = "average"
    single = "single"


class HierarchicalClustering(Clustering):
    """Base class for hierarchical clustering."""

    def __init__(self):
        """Init HierarchicalClustering."""
        super().__init__()
        self.n_clusters: Optional[int] = None
        self.linkage: Optional[str] = None
        self.clustering_algo: Optional[AgglomerativeClustering] = None
        self.distance_matrix: Optional[DistanceMatrix] = None
        self.clusters: Optional[List[int]] = None
        self.df: Optional[pd.DataFrame] = None
        self.segment2cluster: Optional[Dict[str, int]] = None
        self.distance: Optional[Distance] = None
        self.centroids_df: Optional[pd.DataFrame] = None

    def build_distance_matrix(self, df: pd.DataFrame, distance: Distance):
        """Compute distance matrix with given df and distance.

        Parameters
        ----------
        df: pd.DataFrame
            dataframe with series to build distance matrix
        distance: Distance
            instance if distance to compute matrix
        """
        self.df = df
        self.distance = distance
        self.distance_matrix = DistanceMatrix(distance=distance)
        self.distance_matrix.fit(df=df)
        self.clusters = None
        self.segment2cluster = None
        self.centroids_df = None

    def build_clustering_algo(
        self, n_clusters: Optional[int] = 30, linkage: str = ClusteringLinkageMode.average, **clustering_algo_params
    ):
        """Build clustering algo (sklearn.cluster.AgglomerativeClustering) with given params.

        Parameters
        ----------
        n_clusters: int
            number of clusters to build
        linkage: str
            rule for distance computation for new clusters, allowed "ward", "single", "average", "maximum", "complete"

        Notes
        -----
        Note that it will reset previous results of clustering in case of reinit algo.
        """
        self.n_clusters = n_clusters
        self.linkage = linkage
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
        segment-cluster dict: dict
            dict in format {segment: cluster}
        """
        self.clusters = self.clustering_algo.fit_predict(X=self.distance_matrix.matrix)
        self.segment2cluster = {
            self.distance_matrix.idx2segment[i]: self.clusters[i] for i in range(len(self.clusters))
        }
        return self.segment2cluster

    def _get_centroid_for_cluster(self, cluster: str, **averaging_kwargs) -> pd.DataFrame:
        tmp = self.df[self.df["cluster"] == cluster]
        centroid = self.distance.get_average(xs=tmp, **averaging_kwargs)
        centroid["cluster"] = cluster
        return centroid

    def get_centroids(self, **averaging_kwargs) -> pd.DataFrame:
        """Get centroids of clusters.

        Returns
        -------
        centroids_df:
            pd.DataFrame with "cluster", "timestamp", "target" columns
        """
        self.df["cluster"] = self.df["segment"].apply(lambda x: self.segment2cluster[x])
        clusters = self.df["cluster"].unique()
        centroids = []
        for cluster in clusters:
            centroids.append(self._get_centroid_for_cluster(cluster=cluster, **averaging_kwargs))
        self.centroids_df = pd.concat(centroids, ignore_index=True)
        self.df.drop(columns=["cluster"], inplace=True)
        return self.centroids_df


__all__ = ["HierarchicalClustering", "ClusteringLinkageMode"]
