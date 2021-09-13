import numpy as np
import pytest

from etna.transforms import MedianOutliersTransform
from etna.transforms import DensityOutliersTransform
from etna.analysis import get_anomalies_median
from etna.analysis import get_anomalies_density
from etna.datasets.tsdataset import TSDataset


def test_median_outliers_interface():
    """Checks that MedianOutliersTransform doesn't change structure of dataframe."""
    pass


def test_median_outliers_detection():
    """Checks that MedianOutliersTransform detect anomalies according to `get_anomalies_median`."""
    pass
