import numpy as np
import pandas as pd
import pytest

from etna.analysis import get_anomalies_confidence_interval
from etna.analysis import get_anomalies_density
from etna.analysis import get_anomalies_median
from etna.analysis import get_sequence_anomalies
from etna.datasets.tsdataset import TSDataset
from etna.models import ProphetModel
from etna.transforms import ConfidenceIntervalOutliersTransform
from etna.transforms import DensityOutliersTransform
from etna.transforms import MedianOutliersTransform
from etna.transforms import SAXOutliersTransform


@pytest.mark.parametrize(
    "transform",
    [
        MedianOutliersTransform(in_column="target"),
        DensityOutliersTransform(in_column="target"),
        SAXOutliersTransform(in_column="target"),
        ConfidenceIntervalOutliersTransform(in_column="target", model=ProphetModel),
    ],
)
def test_interface(transform, example_tsds: TSDataset):
    """Checks outliers transforms doesn't change structure of dataframe."""
    start_columnns = example_tsds.columns
    example_tsds.fit_transform(transforms=[transform])
    assert np.all(start_columnns == example_tsds.columns)


@pytest.mark.parametrize(
    "transform, method, method_kwargs",
    [
        (MedianOutliersTransform(in_column="target"), get_anomalies_median, {}),
        (DensityOutliersTransform(in_column="target"), get_anomalies_density, {}),
        (SAXOutliersTransform(in_column="target"), get_sequence_anomalies, {}),
        (
            ConfidenceIntervalOutliersTransform(in_column="target", model=ProphetModel),
            get_anomalies_confidence_interval,
            {"model": ProphetModel},
        ),
    ],
)
def test_outliers_detection(transform, method, outliers_tsds, method_kwargs):
    """Checks that outliers transforms detect anomalies according to methods from etna.analysis."""
    detectiom_method_results = method(outliers_tsds, **method_kwargs)

    # save for each segment index without existing nans
    non_nan_index = {}
    for segment in outliers_tsds.segments:
        non_nan_index[segment] = outliers_tsds[:, segment, "target"].dropna().index
    # convert to df to ignore different lengths of series
    transformed_df = transform.fit_transform(outliers_tsds.to_pandas())
    for segment in outliers_tsds.segments:
        nan_timestamps = detectiom_method_results[segment]
        transformed_column = transformed_df.loc[non_nan_index[segment], pd.IndexSlice[segment, "target"]]
        assert np.all(transformed_column[transformed_column.isna()].index == nan_timestamps)
