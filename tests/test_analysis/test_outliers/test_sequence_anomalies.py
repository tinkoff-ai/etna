from typing import List
from typing import Tuple

import numpy as np
import pytest

from etna.analysis.outliers.sequence_outliers import get_segment_sequence_anomalies
from etna.analysis.outliers.sequence_outliers import get_sequence_anomalies
from etna.datasets.tsdataset import TSDataset


@pytest.fixture
def test_sequence_anomalies_interface(outliers_tsds: TSDataset):
    anomaly_seq_dict = get_sequence_anomalies(ts=outliers_tsds, num_anomalies=1, anomaly_lenght=5)

    for segment in ["1", "2"]:
        assert segment in anomaly_seq_dict
        assert isinstance(anomaly_seq_dict[segment], list)
        for pair in anomaly_seq_dict[segment]:
            assert isinstance(pair, tuple)
            assert len(pair) == 2
            assert isinstance(pair[0], np.datetime64) and isinstance(pair[1], np.datetime64)


@pytest.mark.parametrize(
    "arr, expected",
    (
        ([1, 1, 10, -7, 1, 1, 1, 1], [(1, 4)]),
        ([1, 1, 12, 15, 1, 1, 1], [(0, 3)]),
        ([1, 1, -12, -15, 1, 1, 1, 12, 15], [(5, 8), (0, 3)]),
    ),
)
def test_segment_sequence_anomalies(arr: List[int], expected: List[Tuple[int, int]]):
    arr = np.array(arr)
    anomaly_lenght = 3
    num_anomalies = len(expected)
    expected = sorted(expected)

    result = get_segment_sequence_anomalies(series=arr, num_anomalies=num_anomalies, anomaly_lenght=3)
    result = sorted(result)
    for idx in range(num_anomalies):
        assert (result[idx][0] == expected[idx][0]) and (result[idx][1] == expected[idx][1])


@pytest.fixture
def test_sequence_anomalies(outliers_tsds: TSDataset):
    test_sequence_anomalies_interface(tsds)
    expected = {
        "1": [(np.datetime64("2021-01-01"), np.datetime64("2021-01-16"))],
        "2": [(np.datetime64("2021-01-17"), np.datetime64("2021-02-01"))],
    }
    for segment in expected:
        assert expected[segment][0] == anomaly_seq_dict[segment][0]
