import numpy as np
import pandas as pd
import pytest

from etna.clustering.distances.distance_matrix import DistanceMatrix
from etna.clustering.distances.dtw_distance import DTWDistance
from etna.clustering.distances.euclidean_distance import EuclideanDistance


@pytest.fixture
def simple_multisegment_df() -> pd.DataFrame:
    """Generate simple dataframe with multiple segments."""
    date_range = pd.date_range("2020-01-01", periods=4)
    x1 = pd.DataFrame({"timestamp": date_range})
    x1["segment"] = "A"
    x1["target"] = [1, 0, 0, 0]

    x2 = pd.DataFrame({"timestamp": date_range})
    x2["segment"] = "B"
    x2["target"] = [1, 1, 0, 0]

    x3 = pd.DataFrame({"timestamp": date_range})
    x3["segment"] = "C"
    x3["target"] = [0, 1, 0, 0]

    x4 = pd.DataFrame({"timestamp": date_range})
    x4["segment"] = "D"
    x4["target"] = [0, 1, 0, 1]

    df = pd.concat((x1, x2, x3, x4), ignore_index=True)
    df["target"] = df["target"].astype(float)
    return df


def test_idx2segment_segment2idx(simple_multisegment_df: pd.DataFrame):
    """Check that mapping is correct."""
    dm = DistanceMatrix(distance=EuclideanDistance())
    dm.fit(df=simple_multisegment_df)
    assert dm.idx2segment == {0: "A", 1: "B", 2: "C", 3: "D"}
    assert dm.segment2idx == {"A": 0, "B": 1, "C": 2, "D": 3}


def test_eucl_matrix_value(simple_multisegment_df: pd.DataFrame):
    """Check distance matrix in case of euclidean distance."""
    dm = DistanceMatrix(distance=EuclideanDistance())
    dm.fit(df=simple_multisegment_df)
    matrix = dm.predict()

    sqrt_2 = np.sqrt(2)
    sqrt_3 = np.sqrt(3)
    expected = np.array([[0, 1, sqrt_2, sqrt_3], [1, 0, 1, sqrt_2], [sqrt_2, 1, 0, 1], [sqrt_3, sqrt_2, 1, 0]])
    np.testing.assert_array_almost_equal(matrix, expected)


def test_dtw_matrix_value(simple_multisegment_df: pd.DataFrame):
    """Check distance matrix in case of dtw distance."""
    dm = DistanceMatrix(distance=DTWDistance())
    dm.fit(df=simple_multisegment_df)
    matrix = dm.predict()

    expected = np.array([[0, 0, 1, 2], [0, 0, 1, 2], [1, 1, 0, 1], [2, 2, 1, 0]])
    np.testing.assert_array_almost_equal(matrix, expected)
