import numpy as np
import pandas as pd
import pytest

from etna.analysis import get_anomalies_density
from etna.analysis import get_anomalies_median
from etna.analysis import get_anomalies_prediction_interval
from etna.datasets.tsdataset import TSDataset
from etna.models import ProphetModel
from etna.transforms import DensityOutliersTransform
from etna.transforms import MedianOutliersTransform
from etna.transforms import PredictionIntervalOutliersTransform


@pytest.fixture()
def outliers_solid_tsds():
    """Create TSDataset with outliers and same last date."""
    timestamp = pd.date_range("2021-01-01", end="2021-02-20", freq="D")
    target1 = [np.sin(i) for i in range(len(timestamp))]
    target1[10] += 10

    target2 = [np.sin(i) for i in range(len(timestamp))]
    target2[8] += 8
    target2[15] = 2
    target2[26] -= 12

    df1 = pd.DataFrame({"timestamp": timestamp, "target": target1, "segment": "1"})
    df2 = pd.DataFrame({"timestamp": timestamp, "target": target2, "segment": "2"})
    df = pd.concat([df1, df2], ignore_index=True)
    df_exog = df.copy()
    df_exog.columns = ["timestamp", "regressor_1", "segment"]
    ts = TSDataset(
        df=TSDataset.to_dataset(df).iloc[:-10],
        df_exog=TSDataset.to_dataset(df_exog),
        freq="D",
        known_future="all",
    )
    return ts


@pytest.mark.parametrize("in_column", ["target", "regressor_1"])
@pytest.mark.parametrize(
    "transform_constructor, constructor_kwargs",
    [
        (MedianOutliersTransform, {}),
        (DensityOutliersTransform, {}),
        (PredictionIntervalOutliersTransform, dict(model=ProphetModel)),
    ],
)
def test_interface(transform_constructor, constructor_kwargs, outliers_solid_tsds: TSDataset, in_column):
    """Checks outliers transforms doesn't change structure of dataframe."""
    transform = transform_constructor(in_column=in_column, **constructor_kwargs)
    start_columns = outliers_solid_tsds.columns
    outliers_solid_tsds.fit_transform(transforms=[transform])
    assert np.all(start_columns == outliers_solid_tsds.columns)


@pytest.mark.parametrize("in_column", ["target", "exog"])
@pytest.mark.parametrize(
    "transform_constructor, constructor_kwargs, method, method_kwargs",
    [
        (MedianOutliersTransform, {}, get_anomalies_median, {}),
        (DensityOutliersTransform, {}, get_anomalies_density, {}),
        (
            PredictionIntervalOutliersTransform,
            dict(model=ProphetModel),
            get_anomalies_prediction_interval,
            dict(model=ProphetModel),
        ),
    ],
)
def test_outliers_detection(transform_constructor, constructor_kwargs, method, outliers_tsds, method_kwargs, in_column):
    """Checks that outliers transforms detect anomalies according to methods from etna.analysis."""
    detection_method_results = method(outliers_tsds, in_column=in_column, **method_kwargs)
    transform = transform_constructor(in_column=in_column, **constructor_kwargs)

    # save for each segment index without existing nans
    non_nan_index = {}
    for segment in outliers_tsds.segments:
        non_nan_index[segment] = outliers_tsds[:, segment, in_column].dropna().index
    # convert to df to ignore different lengths of series
    transformed_df = transform.fit_transform(outliers_tsds.to_pandas())
    for segment in outliers_tsds.segments:
        nan_timestamps = detection_method_results[segment]
        transformed_column = transformed_df.loc[non_nan_index[segment], pd.IndexSlice[segment, in_column]]
        assert np.all(transformed_column[transformed_column.isna()].index == nan_timestamps)


@pytest.mark.parametrize("in_column", ["target", "regressor_1"])
@pytest.mark.parametrize(
    "transform_constructor, constructor_kwargs",
    [
        (MedianOutliersTransform, {}),
        (DensityOutliersTransform, {}),
        (PredictionIntervalOutliersTransform, dict(model=ProphetModel)),
    ],
)
def test_inverse_transform_train(transform_constructor, constructor_kwargs, outliers_solid_tsds, in_column):
    """Checks that inverse transform returns dataset to its original form."""
    transform = transform_constructor(in_column=in_column, **constructor_kwargs)
    original_df = outliers_solid_tsds.df.copy()
    outliers_solid_tsds.fit_transform([transform])
    outliers_solid_tsds.inverse_transform()

    assert np.all(original_df == outliers_solid_tsds.df)


@pytest.mark.parametrize("in_column", ["target", "regressor_1"])
@pytest.mark.parametrize(
    "transform_constructor, constructor_kwargs",
    [
        (MedianOutliersTransform, {}),
        (DensityOutliersTransform, {}),
        (PredictionIntervalOutliersTransform, dict(model=ProphetModel)),
    ],
)
def test_inverse_transform_future(transform_constructor, constructor_kwargs, outliers_solid_tsds, in_column):
    """Checks that inverse transform does not change the future."""
    transform = transform_constructor(in_column=in_column, **constructor_kwargs)
    outliers_solid_tsds.fit_transform([transform])
    future = outliers_solid_tsds.make_future(future_steps=10)
    original_future_df = future.df.copy()
    future.inverse_transform()
    # check equals and has nans in the same places
    assert np.all((future.df == original_future_df) | (future.df.isna() & original_future_df.isna()))


@pytest.mark.parametrize(
    "transform",
    (
        MedianOutliersTransform(in_column="target"),
        DensityOutliersTransform(in_column="target"),
        PredictionIntervalOutliersTransform(in_column="target", model=ProphetModel),
    ),
)
def test_transform_raise_error_if_not_fitted(transform, outliers_solid_tsds):
    """Test that transform for one segment raise error when calling transform without being fit."""
    with pytest.raises(ValueError, match="Transform is not fitted!"):
        _ = transform.transform(df=outliers_solid_tsds.df)


@pytest.mark.parametrize(
    "transform",
    (
        MedianOutliersTransform(in_column="target"),
        DensityOutliersTransform(in_column="target"),
        PredictionIntervalOutliersTransform(in_column="target", model=ProphetModel),
    ),
)
def test_inverse_transform_raise_error_if_not_fitted(transform, outliers_solid_tsds):
    """Test that transform for one segment raise error when calling inverse_transform without being fit."""
    with pytest.raises(ValueError, match="Transform is not fitted!"):
        _ = transform.inverse_transform(df=outliers_solid_tsds.df)


@pytest.mark.parametrize(
    "transform",
    (
        MedianOutliersTransform(in_column="target"),
        DensityOutliersTransform(in_column="target"),
        PredictionIntervalOutliersTransform(in_column="target", model=ProphetModel),
    ),
)
def test_fit_transform_with_nans(transform, ts_diff_endings):
    ts_diff_endings.fit_transform([transform])
