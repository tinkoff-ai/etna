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


@pytest.fixture()
def outliers_solid_tsds():
    """Create TSDataset with outliers and same last date."""
    timestamp1 = np.arange(np.datetime64("2021-01-01"), np.datetime64("2021-02-10"))
    target1 = [np.sin(i) for i in range(len(timestamp1))]
    target1[10] += 10

    timestamp2 = np.arange(np.datetime64("2021-01-01"), np.datetime64("2021-02-10"))
    target2 = [np.sin(i) for i in range(len(timestamp2))]
    target2[8] += 8
    target2[15] = 2
    target2[26] -= 12

    df1 = pd.DataFrame({"timestamp": timestamp1, "target": target1, "segment": "1"})
    df2 = pd.DataFrame({"timestamp": timestamp2, "target": target2, "segment": "2"})

    df = pd.concat([df1, df2], ignore_index=True)

    df = df.pivot(index="timestamp", columns="segment")
    df = df.reorder_levels([1, 0], axis=1)
    df = df.sort_index(axis=1)
    df.columns.names = ["segment", "feature"]
    tsds = TSDataset(df, "1d")
    return tsds


@pytest.mark.parametrize(
    "transform",
    [
        MedianOutliersTransform(in_column="target"),
        DensityOutliersTransform(in_column="target"),
        SAXOutliersTransform(in_column="target"),
        ConfidenceIntervalOutliersTransform(model=ProphetModel),
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
            ConfidenceIntervalOutliersTransform(model=ProphetModel),
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


@pytest.mark.parametrize(
    "transform",
    [
        MedianOutliersTransform(in_column="target"),
        DensityOutliersTransform(in_column="target"),
        SAXOutliersTransform(in_column="target"),
        ConfidenceIntervalOutliersTransform(model=ProphetModel),
    ],
)
def test_inverse_transform_train(transform, outliers_solid_tsds):
    """Checks that inverse transform returns dataset to its original form."""
    original_df = outliers_solid_tsds.df.copy()
    outliers_solid_tsds.fit_transform([transform])
    outliers_solid_tsds.inverse_transform()

    assert (original_df == outliers_solid_tsds.df).all().all()


@pytest.mark.parametrize(
    "transform",
    [
        MedianOutliersTransform(in_column="target"),
        DensityOutliersTransform(in_column="target"),
        SAXOutliersTransform(in_column="target"),
        ConfidenceIntervalOutliersTransform(model=ProphetModel),
    ],
)
def test_inverse_transform_future(transform, outliers_solid_tsds):
    """Checks that inverse transform does not change the future."""
    outliers_solid_tsds.fit_transform([transform])
    future = outliers_solid_tsds.make_future(future_steps=10)
    original_future_df = future.df.copy()
    future.inverse_transform()

    assert (future.df.isnull() == original_future_df.isnull()).all().all()
