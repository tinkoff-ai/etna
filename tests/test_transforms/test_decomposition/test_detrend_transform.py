from typing import Union

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
from tests.test_transforms.utils import assert_sampling_is_valid
from tests.test_transforms.utils import assert_transformation_equals_loaded_original

DEFAULT_SEGMENT = "segment_1"
TREND_TRANSFORM_TYPE = Union[LinearTrendTransform, TheilSenTrendTransform]


@pytest.fixture
def df_one_segment(example_df) -> pd.DataFrame:
    return example_df[example_df["segment"] == DEFAULT_SEGMENT].set_index("timestamp")


@pytest.fixture
def ts_two_segments(example_df) -> TSDataset:
    df = TSDataset.to_dataset(example_df)
    ts = TSDataset(df=df, freq="D")
    return ts


@pytest.fixture
def ts_two_segments_diff_size(example_df) -> TSDataset:
    df = TSDataset.to_dataset(example_df)
    df.loc[:4, pd.IndexSlice[DEFAULT_SEGMENT, "target"]] = None
    ts = TSDataset(df=df, freq="D")
    return ts


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
def ts_two_segments_linear(df_quadratic) -> TSDataset:
    df_linear = df_quadratic[df_quadratic["segment"].isin(["segment_1", "segment_2"])]
    df = TSDataset.to_dataset(df_linear)
    ts = TSDataset(df=df, freq="D")
    return ts


@pytest.fixture
def df_one_segment_quadratic(df_quadratic) -> pd.DataFrame:
    return df_quadratic[df_quadratic["segment"] == "segment_3"].set_index("timestamp")


@pytest.fixture
def ts_two_segments_quadratic(df_quadratic) -> TSDataset:
    df = TSDataset.to_dataset(df_quadratic)
    ts = TSDataset(df=df, freq="D")
    return ts


def _test_unbiased_fit_transform_one_segment(
    trend_transform: _OneSegmentLinearTrendBaseTransform, df: pd.DataFrame, **comparison_kwargs
) -> None:
    """
    Test if mean of residue after trend subtraction is close to zero in one segment.
    """
    residue = trend_transform.fit_transform(df)["target"].mean()
    npt.assert_almost_equal(residue, 0, **comparison_kwargs)


def _test_unbiased_fit_transform_many_segments(trend_transform, ts: TSDataset, **comparison_kwargs) -> None:
    """
    Test if mean of residue after trend subtraction is close to zero in all segments.
    """
    residue = trend_transform.fit_transform(ts).to_pandas()
    for segment in ts.segments:
        npt.assert_almost_equal(residue[segment, "target"].mean(), 0, **comparison_kwargs)


def _test_fit_transform_one_segment(
    trend_transform: _OneSegmentLinearTrendBaseTransform, df: pd.DataFrame, **comparison_kwargs
) -> None:
    """
    Test if residue after trend subtraction is close to zero in one segment.
    """
    residue = trend_transform.fit_transform(df)["target"]
    residue = residue[~np.isnan(residue)]
    npt.assert_allclose(residue, 0, **comparison_kwargs)


def _test_fit_transform_many_segments(trend_transform, ts: TSDataset, **comparison_kwargs) -> None:
    """
    Test if residue after trend subtraction is close to zero in all segments.
    """
    residue = trend_transform.fit_transform(ts).to_pandas()
    for segment in ts.segments:
        segment_residue = residue[segment, "target"]
        segment_residue = segment_residue[~np.isnan(segment_residue)]
        npt.assert_allclose(segment_residue, 0, **comparison_kwargs)


def test_unbiased_fit_transform_linear_trend_one_segment(df_one_segment: pd.DataFrame) -> None:
    """
    This test checks that LinearRegression predicts unbiased trend on one segment of slightly noised data.
    """
    trend_transform = _OneSegmentLinearTrendBaseTransform(in_column="target", regressor=LinearRegression())
    _test_unbiased_fit_transform_one_segment(trend_transform=trend_transform, df=df_one_segment)


def test_unbiased_fit_transform_theil_sen_trend_one_segment(df_one_segment: pd.DataFrame) -> None:
    """
    This test checks that TheilSenRegressor predicts unbiased trend on one segment of slightly noised data.
    """
    trend_transform = _OneSegmentLinearTrendBaseTransform(
        in_column="target",
        regressor=TheilSenRegressor(n_subsamples=int(len(df_one_segment) / 2), max_iter=3000, tol=1e-4),
    )
    _test_unbiased_fit_transform_one_segment(trend_transform=trend_transform, df=df_one_segment, decimal=0)


def test_unbiased_fit_transform_theil_sen_trend_all_data_one_segment(df_one_segment: pd.DataFrame) -> None:
    """
    This test checks that TheilSenRegressor predicts unbiased trend on one segment of slightly noised data
    using all the data to train model.
    """
    # Note that it is a corner case: we use all the data to predict trend
    trend_transform = _OneSegmentLinearTrendBaseTransform(
        in_column="target", regressor=TheilSenRegressor(n_subsamples=len(df_one_segment))
    )
    _test_unbiased_fit_transform_one_segment(trend_transform=trend_transform, df=df_one_segment)


def test_unbiased_fit_transform_linear_trend_two_segments(ts_two_segments: TSDataset) -> None:
    """
    This test checks that LinearRegression predicts unbiased trend on two segments of slightly noised data.
    """
    trend_transform = LinearTrendTransform(in_column="target")
    _test_unbiased_fit_transform_many_segments(trend_transform=trend_transform, ts=ts_two_segments)


def test_unbiased_fit_transform_theil_sen_trend_two_segments(ts_two_segments: TSDataset) -> None:
    """
    This test checks that TheilSenRegressor predicts unbiased trend on two segments of slightly noised data.
    """
    trend_transform = TheilSenTrendTransform(
        in_column="target", n_subsamples=int(len(ts_two_segments.index) / 2), max_iter=3000, tol=1e-4
    )
    _test_unbiased_fit_transform_many_segments(trend_transform=trend_transform, ts=ts_two_segments, decimal=0)


def test_unbiased_fit_transform_theil_sen_trend_all_data_two_segments(ts_two_segments: TSDataset) -> None:
    """
    This test checks that TheilSenRegressor predicts unbiased trend on two segments of slightly noised data
    using all the data to train model.
    """
    # Note that it is a corner case: we use all the data to predict trend
    trend_transform = TheilSenTrendTransform(in_column="target", n_subsamples=len(ts_two_segments.index))
    _test_unbiased_fit_transform_many_segments(trend_transform=trend_transform, ts=ts_two_segments)


@pytest.mark.parametrize("df_fixture, poly_degree", [("df_one_segment_linear", 1), ("df_one_segment_quadratic", 2)])
def test_fit_transform_linear_trend_one_segment(df_fixture, poly_degree, request) -> None:
    """
    Test that LinearRegression predicts correct trend on one segment of slightly noised data.
    """
    df = request.getfixturevalue(df_fixture)
    trend_transform = _OneSegmentLinearTrendBaseTransform(
        in_column="target", regressor=LinearRegression(), poly_degree=poly_degree
    )
    _test_fit_transform_one_segment(trend_transform=trend_transform, df=df, atol=1e-5)


@pytest.mark.parametrize("df_fixture, poly_degree", [("df_one_segment_linear", 1), ("df_one_segment_quadratic", 2)])
def test_fit_transform_theil_sen_trend_one_segment(df_fixture, poly_degree, request) -> None:
    """
    Test that TheilSenRegressor predicts correct trend on one segment of slightly noised data.

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
    """
    Test that TheilSenRegressor predicts correct trend on one segment of slightly noised data.

    All data is used to train the model.
    """
    df = request.getfixturevalue(df_fixture)
    # Note that it is a corner case: we use all the data to predict trend
    trend_transform = _OneSegmentLinearTrendBaseTransform(
        in_column="target", regressor=TheilSenRegressor(n_subsamples=len(df)), poly_degree=poly_degree
    )
    _test_fit_transform_one_segment(trend_transform=trend_transform, df=df, atol=1e-5)


@pytest.mark.parametrize("ts_fixture, poly_degree", [("ts_two_segments_linear", 1), ("ts_two_segments_quadratic", 2)])
def test_fit_transform_linear_trend_two_segments(ts_fixture, poly_degree, request) -> None:
    """
    Test that LinearRegression predicts correct trend on two segments of slightly noised data.
    """
    ts = request.getfixturevalue(ts_fixture)
    trend_transform = LinearTrendTransform(in_column="target", poly_degree=poly_degree)
    _test_fit_transform_many_segments(trend_transform=trend_transform, ts=ts, atol=1e-5)


@pytest.mark.parametrize("ts_fixture, poly_degree", [("ts_two_segments_linear", 1), ("ts_two_segments_quadratic", 2)])
def test_fit_transform_theil_sen_trend_two_segments(ts_fixture, poly_degree, request) -> None:
    """
    Test that TheilSenRegressor predicts correct trend on two segments of slightly noised data.

    Not all data is used to train the model.
    """
    ts = request.getfixturevalue(ts_fixture)
    trend_transform = TheilSenTrendTransform(
        in_column="target", poly_degree=poly_degree, n_subsamples=int(len(ts.index) / 2), max_iter=3000, tol=1e-4
    )
    _test_fit_transform_many_segments(trend_transform=trend_transform, ts=ts, atol=1e-5)


@pytest.mark.parametrize("ts_fixture, poly_degree", [("ts_two_segments_linear", 1), ("ts_two_segments_quadratic", 2)])
def test_fit_transform_theil_sen_trend_all_data_two_segments(ts_fixture, poly_degree, request) -> None:
    """
    Test that TheilSenRegressor predicts correct trend on two segments of slightly noised data.

    All data is used to train the model.
    """
    ts = request.getfixturevalue(ts_fixture)
    # Note that it is a corner case: we use all the data to predict trend
    trend_transform = TheilSenTrendTransform(in_column="target", poly_degree=poly_degree, n_subsamples=len(ts.index))
    _test_fit_transform_many_segments(trend_transform=trend_transform, ts=ts, atol=1e-5)


def _test_inverse_transform_one_segment(
    trend_transform: _OneSegmentLinearTrendBaseTransform, df: pd.DataFrame, **comparison_kwargs
) -> None:
    """
    Test that trend_transform can correctly make inverse_transform in one segment.
    """
    df_transformed = trend_transform.fit_transform(df)
    df_inverse_transformed = trend_transform.inverse_transform(df_transformed)
    npt.assert_allclose(df["target"], df_inverse_transformed["target"], **comparison_kwargs)


def _test_inverse_transform_many_segments(trend_transform, ts: TSDataset, **comparison_kwargs) -> None:
    """
    Test that trend_transform can correctly make inverse_transform in all segments.
    """
    trend_transform.fit_transform(ts)
    df_inverse_transformed = trend_transform.inverse_transform(ts).to_pandas()
    for segment in ts.segments:
        npt.assert_allclose(
            df_inverse_transformed[segment, "target"], ts.to_pandas()[segment, "target"], **comparison_kwargs
        )


@pytest.mark.parametrize("poly_degree", [1, 2])
def test_inverse_transform_linear_trend_one_segment(df_one_segment: pd.DataFrame, poly_degree: int):
    """
    Test that LinearTrend can correctly make inverse_transform for one segment.
    """
    trend_transform = _OneSegmentLinearTrendBaseTransform(
        in_column="target", regressor=LinearRegression(), poly_degree=poly_degree
    )
    _test_inverse_transform_one_segment(trend_transform=trend_transform, df=df_one_segment)


@pytest.mark.parametrize("poly_degree", [1, 2])
def test_inverse_transform_theil_sen_trend_one_segment(df_one_segment: pd.DataFrame, poly_degree: int):
    """
    Test that TheilSenRegressor can correctly make inverse_transform for one segment.
    """
    trend_transform = _OneSegmentLinearTrendBaseTransform(
        in_column="target", regressor=TheilSenRegressor(n_subsamples=len(df_one_segment)), poly_degree=poly_degree
    )
    _test_inverse_transform_one_segment(trend_transform=trend_transform, df=df_one_segment)


@pytest.mark.parametrize("poly_degree", [1, 2])
def test_inverse_transform_linear_trend_two_segments(ts_two_segments: TSDataset, poly_degree: int):
    """
    Test that LinearTrend can correctly make inverse_transform for two segments.
    """
    trend_transform = LinearTrendTransform(in_column="target", poly_degree=poly_degree)
    _test_inverse_transform_many_segments(trend_transform=trend_transform, ts=ts_two_segments)


@pytest.mark.parametrize("poly_degree", [1, 2])
def test_inverse_transform_theil_sen_trend_two_segments(ts_two_segments: TSDataset, poly_degree: int):
    """
    Test that TheilSenRegressor can correctly make inverse_transform for two segments.
    """
    trend_transform = TheilSenTrendTransform(
        in_column="target", poly_degree=poly_degree, n_subsamples=len(ts_two_segments.index)
    )
    _test_inverse_transform_many_segments(trend_transform=trend_transform, ts=ts_two_segments)


@pytest.mark.parametrize(
    "transformer,decimal",
    [(LinearTrendTransform(in_column="target"), 7), (TheilSenTrendTransform(in_column="target"), 0)],
)
def test_fit_transform_two_segments_diff_size(
    ts_two_segments_diff_size: TSDataset, transformer: TREND_TRANSFORM_TYPE, decimal: int
):
    """
    Test that TrendTransform can correctly make fit_transform for two segments of different size.
    """
    _test_unbiased_fit_transform_many_segments(
        trend_transform=transformer, ts=ts_two_segments_diff_size, decimal=decimal
    )


@pytest.mark.parametrize(
    "transformer", [LinearTrendTransform(in_column="target"), TheilSenTrendTransform(in_column="target")]
)
def test_inverse_transform_segments_diff_size(ts_two_segments_diff_size: TSDataset, transformer: TREND_TRANSFORM_TYPE):
    """
    Test that TrendTransform can correctly make inverse_transform for two segments of different size.
    """
    _test_inverse_transform_many_segments(trend_transform=transformer, ts=ts_two_segments_diff_size)


@pytest.mark.parametrize(
    "transformer,decimal",
    [(LinearTrendTransform(in_column="target"), 7), (TheilSenTrendTransform(in_column="target"), 0)],
)
def test_fit_transform_with_nans(transformer, ts_with_nans, decimal):
    _test_unbiased_fit_transform_many_segments(trend_transform=transformer, ts=ts_with_nans, decimal=decimal)


@pytest.mark.parametrize(
    "transform",
    [LinearTrendTransform(in_column="target"), TheilSenTrendTransform(in_column="target")],
)
def test_save_load(transform, ts_two_segments_linear):
    ts = ts_two_segments_linear
    assert_transformation_equals_loaded_original(transform=transform, ts=ts)


@pytest.mark.parametrize(
    "transform",
    [LinearTrendTransform(in_column="target"), TheilSenTrendTransform(in_column="target")],
)
def test_params_to_tune(transform, ts_two_segments_linear):
    ts = ts_two_segments_linear
    assert len(transform.params_to_tune()) > 0
    assert_sampling_is_valid(transform=transform, ts=ts)
