from copy import deepcopy

import pandas as pd
import pytest
from ruptures import Binseg
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from etna.datasets.tsdataset import TSDataset
from etna.transforms.decomposition import TrendTransform
from etna.transforms.decomposition.trend import _OneSegmentTrendTransform

DEFAULT_SEGMENT = "segment_1"


@pytest.fixture
def df_one_segment(example_df) -> pd.DataFrame:
    return example_df[example_df["segment"] == DEFAULT_SEGMENT].set_index("timestamp")


def test_fit_transform_one_segment(df_one_segment: pd.DataFrame) -> None:
    """
    Test that fit_transform interface works correctly for one segment.
    """
    df_one_segment_original = df_one_segment.copy()
    out_column = "regressor_result"
    trend_transform = _OneSegmentTrendTransform(
        in_column="target",
        change_point_model=Binseg(),
        detrend_model=LinearRegression(),
        n_bkps=5,
        out_column=out_column,
    )
    df_one_segment = trend_transform.fit_transform(df_one_segment)
    assert sorted(df_one_segment.columns) == sorted(["target", "segment", out_column])
    assert (df_one_segment["target"] == df_one_segment_original["target"]).all()
    residue = df_one_segment["target"] - df_one_segment[out_column]
    assert residue.mean() < 1


def test_inverse_transform_one_segment(df_one_segment: pd.DataFrame) -> None:
    """
    Test that inverse_transform interface works correctly for one segment.
    """
    trend_transform = _OneSegmentTrendTransform(
        in_column="target",
        change_point_model=Binseg(),
        detrend_model=LinearRegression(),
        n_bkps=5,
        out_column="test",
    )
    df_one_segment_transformed = trend_transform.fit_transform(df_one_segment)
    df_one_segment_inverse_transformed = trend_transform.inverse_transform(df_one_segment)
    assert (df_one_segment_transformed == df_one_segment_inverse_transformed).all().all()


def test_fit_transform_many_segments(example_tsds: TSDataset) -> None:
    """
    Test that fit_transform interface works correctly for many segment.
    """
    out_column = "regressor_result"
    example_tsds_original = deepcopy(example_tsds)
    trend_transform = TrendTransform(
        in_column="target",
        detrend_model=LinearRegression(),
        n_bkps=5,
        out_column=out_column,
    )
    example_tsds.fit_transform([trend_transform])
    for segment in example_tsds.segments:
        segment_slice = example_tsds[:, segment, :][segment]
        segment_slice_original = example_tsds_original[:, segment, :][segment]
        assert sorted(segment_slice.columns) == sorted(["target", out_column])
        assert (segment_slice["target"] == segment_slice_original["target"]).all()
        residue = segment_slice_original["target"] - segment_slice[out_column]
        assert residue.mean() < 1


def test_inverse_transform_many_segments(example_tsds: TSDataset) -> None:
    """
    Test that inverse_transform interface works correctly for many segment.
    """
    trend_transform = TrendTransform(
        in_column="target",
        detrend_model=LinearRegression(),
        n_bkps=5,
        out_column="test",
    )
    example_tsds.fit_transform([trend_transform])
    original_df = example_tsds.df.copy()
    example_tsds.inverse_transform()
    assert (original_df == example_tsds.df).all().all()


def test_transform_inverse_transform(example_tsds: TSDataset) -> None:
    """
    Test inverse transform of TrendTransform.
    """
    trend_transform = TrendTransform(in_column="target", detrend_model=LinearRegression(), model="rbf")
    example_tsds.fit_transform([trend_transform])
    original = example_tsds.df.copy()
    example_tsds.inverse_transform()
    assert (example_tsds.df == original).all().all()


def test_transform_interface_out_column(example_tsds: TSDataset) -> None:
    """Test transform interface with out_column param"""
    out_column = "regressor_test"
    trend_transform = TrendTransform(
        in_column="target", detrend_model=LinearRegression(), model="rbf", out_column=out_column
    )
    result = trend_transform.fit_transform(example_tsds.df)
    for seg in result.columns.get_level_values(0).unique():
        assert out_column in result[seg].columns


def test_transform_interface_repr(example_tsds: TSDataset) -> None:
    """Test transform interface without out_column param"""
    trend_transform = TrendTransform(in_column="target", detrend_model=LinearRegression(), model="rbf")
    out_column = f"{trend_transform.__repr__()}"
    result = trend_transform.fit_transform(example_tsds.df)
    for seg in result.columns.get_level_values(0).unique():
        assert out_column in result[seg].columns


@pytest.mark.parametrize("model", (LinearRegression(), RandomForestRegressor()))
def test_fit_transform_with_nans_in_tails(df_with_nans_in_tails, model):
    transform = TrendTransform(in_column="target", detrend_model=model, model="rbf", out_column="regressor_result")
    transformed = transform.fit_transform(df=df_with_nans_in_tails)
    for segment in transformed.columns.get_level_values("segment").unique():
        segment_slice = transformed.loc[pd.IndexSlice[:], pd.IndexSlice[segment, :]][segment]
        residue = segment_slice["target"] - segment_slice["regressor_result"]
        assert residue.mean() < 0.13


@pytest.mark.parametrize("model", (LinearRegression(), RandomForestRegressor()))
def test_fit_transform_with_nans_in_middle_raise_error(df_with_nans, model):
    transform = TrendTransform(in_column="target", detrend_model=model, model="rbf")
    with pytest.raises(ValueError, match="The input column contains NaNs in the middle of the series!"):
        _ = transform.fit_transform(df=df_with_nans)
