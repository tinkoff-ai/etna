import numpy as np
import pytest

from etna.analysis import get_anomalies_density
from etna.analysis import get_anomalies_median
from etna.datasets.tsdataset import TSDataset
from etna.transforms import DensityOutliersTransform
from etna.transforms import MedianOutliersTransform


@pytest.mark.parametrize(
    "transform", [MedianOutliersTransform(in_column="target"), DensityOutliersTransform(in_column="target")]
)
def test_interface(transform, example_tsds: TSDataset):
    """Checks that MedianOutliersTransform and DensityOutliersTransform doesn't change structure of dataframe."""
    start_columnns = example_tsds.columns
    example_tsds.fit_transform(transforms=[transform])
    assert np.all(start_columnns == example_tsds.columns)


@pytest.mark.parametrize(
    "transform, method",
    [
        (MedianOutliersTransform(in_column="target"), get_anomalies_median),
        (DensityOutliersTransform(in_column="target"), get_anomalies_density),
    ],
)
def test_outliers_detection(transform, method, outliers_tsds, recwarn):
    """Checks that MedianOutliersTransform detect anomalies according to `get_anomalies_median`."""
    detectiom_method_results = method(outliers_tsds)

    # save for each segment index without existing nans
    non_nan_index = {}
    for segment in outliers_tsds.segments:
        non_nan_index[segment] = outliers_tsds[:, segment, "target"].dropna().index

    # make transform and compare nans
    outliers_tsds.fit_transform(transforms=[transform])
    for segment in outliers_tsds.segments:
        nan_timestamps = detectiom_method_results[segment]
        transformed_column = outliers_tsds[non_nan_index[segment], segment, "target"]
        assert np.all(transformed_column[transformed_column.isna()].index == nan_timestamps)
