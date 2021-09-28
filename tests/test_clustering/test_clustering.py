import numpy as np
import pandas as pd
import pytest

from etna.clustering import DTWClustering
from etna.clustering import HierarchicalClustering
from etna.clustering.hierarchical.euclidean_clustering import EuclideanClustering


@pytest.fixture
def eucl_df() -> pd.DataFrame:
    df = pd.DataFrame()
    for i in range(1, 8):
        date_range = pd.date_range("2020-01-01", "2020-05-01")
        for j, sigma in enumerate([0.1, 0.3, 0.5, 0.8]):
            tmp = pd.DataFrame({"timestamp": date_range})
            tmp["segment"] = f"{i}{j}"
            tmp["target"] = np.random.normal(i, sigma, len(tmp))
            df = df.append(tmp, ignore_index=True)
    return df


def test_eucl_clustering(eucl_df: pd.DataFrame):
    """Check that all the series are divided to the clusters according to mu
    (in case of number of clusters is equal to number of different mus)."""
    clustering = EuclideanClustering()
    clustering.build_distance_matrix(df=eucl_df)
    clustering.build_clustering_algo(n_clusters=7)
    segment2clusters = clustering.fit_predict()
    assert len(set(segment2clusters.values())) == 7
    eucl_df["cluster"] = eucl_df["segment"].apply(lambda x: clustering.segment2cluster[x])
    eucl_df["expected_mean"] = eucl_df["segment"].apply(lambda x: int(x[0]))
    res = eucl_df.groupby("cluster")["expected_mean"].agg(min="min", max="max", mean="mean").reset_index()
    assert (res["min"] == res["max"]).all()
    assert (res["mean"] == res["max"]).all()


def test_dtw_clustering(eucl_df: pd.DataFrame):
    """Check that dtw clustering works."""
    clustering = DTWClustering()
    clustering.build_distance_matrix(df=eucl_df)
    clustering.build_clustering_algo(n_clusters=3)
    segment2clusters = clustering.fit_predict()
    assert len(set(segment2clusters.values())) == 3


@pytest.mark.parametrize(
    "clustering,n_clusters",
    ((EuclideanClustering(), 5), (EuclideanClustering(), 7), (DTWClustering(), 3), (DTWClustering(), 5)),
)
def test_centroids(eucl_df: pd.DataFrame, clustering: HierarchicalClustering, n_clusters: int):
    """Check that centroids work in euclidean clustering pipeline."""
    clustering.build_distance_matrix(df=eucl_df)
    clustering.build_clustering_algo(n_clusters=n_clusters)
    _ = clustering.fit_predict()
    centroids = clustering.get_centroids()
    assert isinstance(centroids, pd.DataFrame)
    assert sorted(centroids.columns) == ["cluster", "target", "timestamp"]
    assert len(centroids["cluster"].unique()) == n_clusters
