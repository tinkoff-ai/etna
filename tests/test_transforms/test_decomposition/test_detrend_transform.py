import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import TheilSenRegressor

from etna.datasets.tsdataset import TSDataset
from etna.transforms.decomposition import LinearTrendTransform
from etna.transforms.decomposition import TheilSenTrendTransform
from etna.transforms.decomposition.detrend import _OneSegmentLinearTrendBaseTransform


@pytest.fixture
def df_quadratic() -> pd.DataFrame:
    """Make dataframe with quadratic trends. Segments 1, 2 has linear trend, segments -- 3, 4 quadratic."""
    timestamp = pd.date_range(start="2020-01-01", end="2020-02-01", freq="H")
    rng = np.random.default_rng(42)
    df_template = pd.DataFrame({"timestamp": timestamp, "segment": "segment", "target": np.arange(len(timestamp))})

    # create segments
    sigma = 0.05
    df_1 = df_template.copy()
    df_1["target"] = 0.1 * df_1["target"] + rng.normal(scale=sigma)
    df_1["segment"] = "segment_1"

    df_2 = df_template.copy()
    df_2["target"] = (-2) * df_2["target"] + rng.normal(scale=sigma)
    df_2["segment"] = "segment_2"

    df_3 = df_template.copy()
    df_3["target"] = 0.01 * df_3["target"] ** 2 + rng.normal(scale=sigma)
    df_3["segment"] = "segment_3"

    df_4 = df_template.copy()
    df_4["target"] = 0.01 * df_4["target"] ** 2 + 0.1 * df_4["target"] + rng.normal(scale=sigma)
    df_4["segment"] = "segment_4"

    # build final dataframe
    df = pd.concat([df_1, df_2, df_3, df_4], ignore_index=True)
    return df


@pytest.fixture
def df_one_segment_linear(df_quadratic) -> pd.DataFrame:
    return df_quadratic[df_quadratic["segment"] == "segment_1"].set_index("timestamp")


@pytest.fixture
def df_many_segments_linear(df_quadratic) -> pd.DataFrame:
    df_linear = df_quadratic[df_quadratic["segment"].isin(["segment_1", "segment_2"])]
    return TSDataset.to_dataset(df_linear)


@pytest.fixture
def df_one_segment_quadratic(df_quadratic) -> pd.DataFrame:
    return df_quadratic[df_quadratic["segment"] == "segment_3"].set_index("timestamp")


@pytest.fixture
def df_many_segments_quadratic(df_quadratic) -> pd.DataFrame:
    return TSDataset.to_dataset(df_quadratic)


@pytest.fixture
def df_many_segments_diff_size(df_quadratic) -> pd.DataFrame:
    df = TSDataset.to_dataset(df_quadratic)
    df.loc[:4, pd.IndexSlice["segment_1", "target"]] = None
    return df


@pytest.fixture
def df_many_segments_with_nans(df_quadratic) -> pd.DataFrame:
    df = TSDataset.to_dataset(df_quadratic)
    df.loc[:4, pd.IndexSlice["segment_1", "target"]] = None
    df.loc[-3:, pd.IndexSlice["segment_1", "target"]] = None
    df.loc[[df.index[5], df.index[8]], pd.IndexSlice["segment_1", "target"]] = None
    return df


def _test_fit_transform_one_segment(
    trend_transform: _OneSegmentLinearTrendBaseTransform, df: pd.DataFrame, **comparison_kwargs
) -> None:
    """Test if residue after trend subtraction is close to zero in one segment.

    Parameters
    ----------
    trend_transform:
        instance of OneSegmentLinearTrendBaseTransform to predict trend with
    df:
        dataframe to predict
    comparison_kwargs:
        arguments for numpy.testing.assert_allclose function in key-value format
    """
    residue = trend_transform.fit_transform(df)["target"]
    residue = residue[~np.isnan(residue)]
    npt.assert_allclose(residue, 0, **comparison_kwargs)


def _test_fit_transform_many_segments(trend_transform, df: pd.DataFrame, **comparison_kwargs) -> None:
    """Test if residue after trend subtraction is close to zero in all segments.

    Parameters
    ----------
    trend_transform:
         instance of LinearTrendTransform or TheilSenTrendTransform to predict trend with
    df:
        dataframe to predict
    comparison_kwargs:
        arguments for numpy.testing.assert_allclose function in key-value format
    """
    residue = trend_transform.fit_transform(df)
    for segment in df.columns.get_level_values("segment").unique():
        segment_residue = residue[segment, "target"]
        segment_residue = segment_residue[~np.isnan(segment_residue)]
        npt.assert_allclose(segment_residue, 0, **comparison_kwargs)


@pytest.mark.parametrize("df_fixture, poly_degree", [("df_one_segment_linear", 1), ("df_one_segment_quadratic", 2)])
def test_fit_transform_linear_trend_one_segment(df_fixture, poly_degree, request) -> None:
    """Test that LinearRegression predicts correct trend on one segment of slightly noised data."""
    df = request.getfixturevalue(df_fixture)
    trend_transform = _OneSegmentLinearTrendBaseTransform(
        in_column="target", regressor=LinearRegression(), poly_degree=poly_degree
    )
    _test_fit_transform_one_segment(trend_transform=trend_transform, df=df, atol=1e-5)


@pytest.mark.parametrize("df_fixture, poly_degree", [("df_one_segment_linear", 1), ("df_one_segment_quadratic", 2)])
def test_fit_transform_theil_sen_trend_one_segment(df_fixture, poly_degree, request) -> None:
    """Test that TheilSenRegressor predicts correct trend on one segment of slightly noised data.

    Not all data is used to train the model.
    """
    df = request.getfixturevalue(df_fixture)
    trend_transform = _OneSegmentLinearTrendBaseTransform(
        in_column="target",
        regressor=TheilSenRegressor(n_subsamples=int(len(df) / 2), max_iter=3000, tol=1e-4),
        poly_degree=poly_degree,
    )
    _test_fit_transform_one_segment(trend_transform=trend_transform, df=df, atol=1e-5)


@pytest.mark.parametrize("df_fixture, poly_degree", [("df_one_segment_linear", 1), ("df_one_segment_quadratic", 2)])
def test_fit_transform_theil_sen_trend_all_data_one_segment(df_fixture, poly_degree, request) -> None:
    """Test that TheilSenRegressor predicts correct trend on one segment of slightly noised data.

    All data is used to train the model.
    """
    df = request.getfixturevalue(df_fixture)
    # Note that it is a corner case: we use all the data to predict trend
    trend_transform = _OneSegmentLinearTrendBaseTransform(
        in_column="target", regressor=TheilSenRegressor(n_subsamples=len(df)), poly_degree=poly_degree
    )
    _test_fit_transform_one_segment(trend_transform=trend_transform, df=df, atol=1e-5)


@pytest.mark.parametrize("df_fixture, poly_degree", [("df_many_segments_linear", 1), ("df_many_segments_quadratic", 2)])
def test_fit_transform_linear_trend_many_segments(df_fixture, poly_degree, request) -> None:
    """Test that LinearRegression predicts correct trend on many segments of slightly noised data."""
    df = request.getfixturevalue(df_fixture)
    trend_transform = LinearTrendTransform(in_column="target", poly_degree=poly_degree)
    _test_fit_transform_many_segments(trend_transform=trend_transform, df=df, atol=1e-5)


@pytest.mark.parametrize("df_fixture, poly_degree", [("df_many_segments_linear", 1), ("df_many_segments_quadratic", 2)])
def test_fit_transform_theil_sen_trend_many_segments(df_fixture, poly_degree, request) -> None:
    """Test that TheilSenRegressor predicts correct trend on many segments of slightly noised data.

    Not all data is used to train the model.
    """
    df = request.getfixturevalue(df_fixture)
    trend_transform = TheilSenTrendTransform(
        in_column="target", poly_degree=poly_degree, n_subsamples=int(len(df) / 2), max_iter=3000, tol=1e-4
    )
    _test_fit_transform_many_segments(trend_transform=trend_transform, df=df, atol=1e-5)


@pytest.mark.parametrize("df_fixture, poly_degree", [("df_many_segments_linear", 1), ("df_many_segments_quadratic", 2)])
def test_fit_transform_theil_sen_trend_all_data_many_segments(df_fixture, poly_degree, request) -> None:
    """Test that TheilSenRegressor predicts correct trend on many segments of slightly noised data.

    All data is used to train the model.
    """
    df = request.getfixturevalue(df_fixture)
    # Note that it is a corner case: we use all the data to predict trend
    trend_transform = TheilSenTrendTransform(in_column="target", poly_degree=poly_degree, n_subsamples=len(df))
    _test_fit_transform_many_segments(trend_transform=trend_transform, df=df, atol=1e-5)


def _test_inverse_transform_one_segment(
    trend_transform: _OneSegmentLinearTrendBaseTransform, df: pd.DataFrame, **comparison_kwargs
) -> None:
    """Test that trend_transform can correctly make inverse_transform in one segment.

    Parameters
    ----------
    trend_transform:
        instance of LinearTrendBaseTransform to predict trend with
    df:
        dataframe to predict
    comparison_kwargs:
        arguments for numpy.testing.assert_allclose function in key-value format
    """
    df_transformed = trend_transform.fit_transform(df)
    df_inverse_transformed = trend_transform.inverse_transform(df_transformed)
    npt.assert_allclose(df["target"], df_inverse_transformed["target"], **comparison_kwargs)


def _test_inverse_transform_many_segments(trend_transform, df: pd.DataFrame, **comparison_kwargs) -> None:
    """Test that trend_transform can correctly make inverse_transform in all segments.

    Parameters
    ----------
    trend_transform:
        instance of LinearTrendTransform or TheilSenTrendTransform to predict trend with
    df:
        dataframe to predict
    comparison_kwargs:
        arguments for numpy.testing.assert_allclose function in key-value format
    """
    df_transformed = trend_transform.fit_transform(df)
    df_inverse_transformed = trend_transform.inverse_transform(df_transformed)
    for segment in df.columns.get_level_values("segment").unique():
        npt.assert_allclose(df_inverse_transformed[segment, "target"], df[segment, "target"], **comparison_kwargs)


@pytest.mark.parametrize("df_fixture, poly_degree", [("df_one_segment_linear", 1), ("df_one_segment_quadratic", 2)])
def test_inverse_transform_linear_trend_one_segment(df_fixture, poly_degree, request):
    """Test that LinearTrend can correctly make inverse_transform for one segment."""
    df = request.getfixturevalue(df_fixture)
    trend_transform = _OneSegmentLinearTrendBaseTransform(
        in_column="target", regressor=LinearRegression(), poly_degree=poly_degree
    )
    _test_inverse_transform_one_segment(trend_transform=trend_transform, df=df)


@pytest.mark.parametrize("df_fixture, poly_degree", [("df_one_segment_linear", 1), ("df_one_segment_quadratic", 2)])
def test_inverse_transform_theil_sen_trend_one_segment(df_fixture, poly_degree, request):
    """Test that TheilSenRegressor can correctly make inverse_transform for one segment."""
    df = request.getfixturevalue(df_fixture)
    trend_transform = _OneSegmentLinearTrendBaseTransform(
        in_column="target", regressor=TheilSenRegressor(n_subsamples=len(df)), poly_degree=poly_degree
    )
    _test_inverse_transform_one_segment(trend_transform=trend_transform, df=df)


@pytest.mark.parametrize("df_fixture, poly_degree", [("df_many_segments_linear", 1), ("df_many_segments_quadratic", 2)])
def test_inverse_transform_linear_trend_many_segments(df_fixture, poly_degree, request):
    """Test that LinearTrend can correctly make inverse_transform for many segments."""
    df = request.getfixturevalue(df_fixture)
    trend_transform = LinearTrendTransform(in_column="target", poly_degree=poly_degree)
    _test_inverse_transform_many_segments(trend_transform=trend_transform, df=df)


@pytest.mark.parametrize("df_fixture, poly_degree", [("df_many_segments_linear", 1), ("df_many_segments_quadratic", 2)])
def test_inverse_transform_theil_sen_trend_many_segments(df_fixture, poly_degree, request):
    """Test that TheilSenRegressor can correctly make inverse_transform for many segments."""
    df = request.getfixturevalue(df_fixture)
    trend_transform = TheilSenTrendTransform(in_column="target", poly_degree=poly_degree, n_subsamples=len(df))
    _test_inverse_transform_many_segments(trend_transform=trend_transform, df=df)


@pytest.mark.parametrize(
    "transformer",
    [
        LinearTrendTransform(in_column="target", poly_degree=2),
        TheilSenTrendTransform(in_column="target", poly_degree=2),
    ],
)
def test_fit_transform_two_segments_diff_size(df_many_segments_diff_size, transformer):
    """Test that TrendTransform can correctly make fit_transform for two segments of different size."""
    _test_fit_transform_many_segments(trend_transform=transformer, df=df_many_segments_diff_size, atol=1e-5)


@pytest.mark.parametrize(
    "transformer",
    [
        LinearTrendTransform(in_column="target", poly_degree=2),
        TheilSenTrendTransform(in_column="target", poly_degree=2),
    ],
)
def test_inverse_transform_segments_diff_size(df_many_segments_diff_size, transformer):
    """Test that TrendTransform can correctly make inverse_transform for two segments of different size."""
    _test_inverse_transform_many_segments(trend_transform=transformer, df=df_many_segments_diff_size)


@pytest.mark.parametrize(
    "transformer",
    [
        LinearTrendTransform(in_column="target", poly_degree=2),
        TheilSenTrendTransform(in_column="target", poly_degree=2),
    ],
)
def test_fit_transform_with_nans(transformer, df_many_segments_with_nans):
    _test_fit_transform_many_segments(trend_transform=transformer, df=df_many_segments_with_nans, atol=1e-5)
