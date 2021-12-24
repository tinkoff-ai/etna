import numpy.testing as npt
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import TheilSenRegressor

from etna.datasets.tsdataset import TSDataset
from etna.transforms.base import PerSegmentWrapper
from etna.transforms.detrend import LinearTrendTransform
from etna.transforms.detrend import TheilSenTrendTransform
from etna.transforms.detrend import _OneSegmentLinearTrendBaseTransform

DEFAULT_SEGMENT = "segment_1"


@pytest.fixture
def df_one_segment(example_df) -> pd.DataFrame:
    return example_df[example_df["segment"] == DEFAULT_SEGMENT].set_index("timestamp")


@pytest.fixture
def df_two_segments(example_df) -> pd.DataFrame:
    return TSDataset.to_dataset(example_df)


@pytest.fixture
def df_two_segments_diff_size(example_df) -> pd.DataFrame:
    df = TSDataset.to_dataset(example_df)
    df.loc[:4, pd.IndexSlice[DEFAULT_SEGMENT, "target"]] = None
    return df


def _test_fit_transform_one_segment(
    trend_transform: _OneSegmentLinearTrendBaseTransform, df: pd.DataFrame, **comparison_kwargs
) -> None:
    """
    Test if residue after trend subtraction is close to zero in one segment.

    Parameters
    ----------
    trend_transform:
        instance of OneSegmentLinearTrendBaseTransform to predict trend with
    df:
        dataframe to predict
    comparison_kwargs:
        arguments for numpy.testing.assert_almost_equal function in key-value format
    """
    residue = trend_transform.fit_transform(df)["target"].mean()
    npt.assert_almost_equal(residue, 0, **comparison_kwargs)


def _test_fit_transform_many_segments(trend_transform, df: pd.DataFrame, **comparison_kwargs) -> None:
    """
    Test if residue after trend subtraction is close to zero in all segments.

    Parameters
    ----------
    trend_transform:
         instance of LinearTrendTransform or TheilSenTrendTransform to predict trend with
    df:
        dataframe to predict
    comparison_kwargs:
        arguments for numpy.testing.assert_almost_equal function in key-value format
    """
    residue = trend_transform.fit_transform(df)
    for segment in df.columns.get_level_values("segment").unique():
        npt.assert_almost_equal(residue[segment, "target"].mean(), 0, **comparison_kwargs)


def test_fit_transform_linear_trend_one_segment(df_one_segment: pd.DataFrame) -> None:
    """
    This test checks that LinearRegression predicts correct trend on one segment of slightly noised data.
    """
    trend_transform = _OneSegmentLinearTrendBaseTransform(in_column="target", regressor=LinearRegression())
    _test_fit_transform_one_segment(trend_transform=trend_transform, df=df_one_segment)


def test_fit_transform_theil_sen_trend_one_segment(df_one_segment: pd.DataFrame) -> None:
    """
    This test checks that TheilSenRegressor predicts correct trend on one segment of slightly noised data.
    """
    trend_transform = _OneSegmentLinearTrendBaseTransform(
        in_column="target",
        regressor=TheilSenRegressor(n_subsamples=int(len(df_one_segment) / 2), max_iter=3000, tol=1e-4),
    )
    _test_fit_transform_one_segment(trend_transform=trend_transform, df=df_one_segment, decimal=0)


def test_fit_transform_theil_sen_trend_all_data_one_segment(df_one_segment: pd.DataFrame) -> None:
    """
    This test checks that TheilSenRegressor predicts correct trend on one segment of slightly noised data
    using all the data to train model.
    """
    # Note that it is a corner case: we use all the data to predict trend
    trend_transform = _OneSegmentLinearTrendBaseTransform(
        in_column="target", regressor=TheilSenRegressor(n_subsamples=len(df_one_segment))
    )
    _test_fit_transform_one_segment(trend_transform=trend_transform, df=df_one_segment)


def test_fit_transform_linear_trend_two_segments(df_two_segments: pd.DataFrame) -> None:
    """
    This test checks that LinearRegression predicts correct trend on two segments of slightly noised data.
    """
    trend_transform = LinearTrendTransform(in_column="target")
    _test_fit_transform_many_segments(trend_transform=trend_transform, df=df_two_segments)


def test_fit_transform_theil_sen_trend_two_segments(df_two_segments: pd.DataFrame) -> None:
    """
    This test checks that TheilSenRegressor predicts correct trend on two segments of slightly noised data.
    """
    trend_transform = TheilSenTrendTransform(
        in_column="target", n_subsamples=int(len(df_two_segments) / 2), max_iter=3000, tol=1e-4
    )
    _test_fit_transform_many_segments(trend_transform=trend_transform, df=df_two_segments, decimal=0)


def test_fit_transform_theil_sen_trend_all_data_two_segments(df_two_segments: pd.DataFrame) -> None:
    """
    This test checks that TheilSenRegressor predicts correct trend on two segments of slightly noised data
    using all the data to train model.
    """
    # Note that it is a corner case: we use all the data to predict trend
    trend_transform = TheilSenTrendTransform(in_column="target", n_subsamples=len(df_two_segments))
    _test_fit_transform_many_segments(trend_transform=trend_transform, df=df_two_segments)


def _test_inverse_transform_one_segment(
    trend_transform: _OneSegmentLinearTrendBaseTransform, df: pd.DataFrame, **comparison_kwargs
) -> None:
    """
    Test that trend_transform can correctly make inverse_transform in one segment.

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
    npt.assert_allclose(df["target"], df_inverse_transformed["target"])


def _test_inverse_transform_many_segments(trend_transform, df: pd.DataFrame, **comparison_kwargs) -> None:
    """
    Test that trend_transform can correctly make inverse_transform in all segments.

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


def test_inverse_transform_linear_trend_one_segment(df_one_segment: pd.DataFrame):
    """
    Test that LinearTrend can correctly make inverse_transform for one segment.
    """
    trend_transform = _OneSegmentLinearTrendBaseTransform(in_column="target", regressor=LinearRegression())
    _test_inverse_transform_one_segment(trend_transform=trend_transform, df=df_one_segment)


def test_inverse_transform_theil_sen_trend_one_segment(df_one_segment: pd.DataFrame):
    """
    Test that TheilSenRegressor can correctly make inverse_transform for one segment.
    """
    trend_transform = _OneSegmentLinearTrendBaseTransform(
        in_column="target", regressor=TheilSenRegressor(n_subsamples=len(df_one_segment))
    )
    _test_inverse_transform_one_segment(trend_transform=trend_transform, df=df_one_segment)


def test_inverse_transform_linear_trend_two_segments(df_two_segments: pd.DataFrame):
    """
    Test that LinearTrend can correctly make inverse_transform for two segments.
    """
    trend_transform = LinearTrendTransform(in_column="target")
    _test_inverse_transform_many_segments(trend_transform=trend_transform, df=df_two_segments)


def test_inverse_transform_theil_sen_trend_two_segments(df_two_segments: pd.DataFrame):
    """
    Test that TheilSenRegressor can correctly make inverse_transform for two segments.
    """
    trend_transform = TheilSenTrendTransform(in_column="target", n_subsamples=len(df_two_segments))
    _test_inverse_transform_many_segments(trend_transform=trend_transform, df=df_two_segments)


@pytest.mark.parametrize(
    "transformer,decimal",
    [(LinearTrendTransform(in_column="target"), 7), (TheilSenTrendTransform(in_column="target"), 0)],
)
def test_fit_transform_two_segments_diff_size(
    df_two_segments_diff_size: pd.DataFrame, transformer: PerSegmentWrapper, decimal: int
):
    """
    Test that TrendTransform can correctly make fit_transform for two segments of different size.
    """
    _test_fit_transform_many_segments(trend_transform=transformer, df=df_two_segments_diff_size, decimal=decimal)


@pytest.mark.parametrize(
    "transformer", [LinearTrendTransform(in_column="target"), TheilSenTrendTransform(in_column="target")]
)
def test_inverse_transform_segments_diff_size(df_two_segments_diff_size: pd.DataFrame, transformer: PerSegmentWrapper):
    """
    Test that TrendTransform can correctly make inverse_transform for two segments of different size.
    """
    _test_inverse_transform_many_segments(trend_transform=transformer, df=df_two_segments_diff_size)


@pytest.xfail
@pytest.mark.parametrize(
    "transformer", [LinearTrendTransform(in_column="target"), TheilSenTrendTransform(in_column="target")]
)
def test_fit_transform_with_nans(transformer, ts_diff_endings):
    ts_diff_endings.fit_transform([transformer])
