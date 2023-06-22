from copy import deepcopy

import pandas as pd
import pytest
from ruptures import Binseg
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from etna.datasets.tsdataset import TSDataset
from etna.transforms.decomposition import TrendTransform
from etna.transforms.decomposition.change_points_based.change_points_models import RupturesChangePointsModel
from etna.transforms.decomposition.change_points_based.per_interval_models import SklearnRegressionPerIntervalModel
from etna.transforms.decomposition.change_points_based.trend import _OneSegmentTrendTransform
from tests.test_transforms.utils import assert_sampling_is_valid
from tests.test_transforms.utils import assert_transformation_equals_loaded_original

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
        change_points_model=RupturesChangePointsModel(change_points_model=Binseg(), n_bkps=5),
        per_interval_model=SklearnRegressionPerIntervalModel(),
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
        change_points_model=RupturesChangePointsModel(change_points_model=Binseg(), n_bkps=5),
        per_interval_model=SklearnRegressionPerIntervalModel(),
        out_column="test",
    )
    df_one_segment_transformed = trend_transform.fit_transform(df_one_segment)
    df_one_segment_inverse_transformed = trend_transform.inverse_transform(df_one_segment)
    pd.testing.assert_frame_equal(df_one_segment_transformed, df_one_segment_inverse_transformed)


def test_fit_transform_many_segments(example_tsds: TSDataset) -> None:
    """
    Test that fit_transform interface works correctly for many segment.
    """
    out_column = "regressor_result"
    example_tsds_original = deepcopy(example_tsds)
    trend_transform = TrendTransform(
        in_column="target",
        per_interval_model=SklearnRegressionPerIntervalModel(),
        change_points_model=RupturesChangePointsModel(change_points_model=Binseg(), n_bkps=5),
        out_column=out_column,
    )
    trend_transform.fit_transform(example_tsds)
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
        per_interval_model=SklearnRegressionPerIntervalModel(),
        change_points_model=RupturesChangePointsModel(change_points_model=Binseg(), n_bkps=5),
        out_column="test",
    )
    trend_transform.fit_transform(example_tsds)
    original_df = example_tsds.to_pandas()
    trend_transform.inverse_transform(example_tsds)
    pd.testing.assert_frame_equal(original_df, example_tsds.to_pandas())


def test_transform_inverse_transform(example_tsds: TSDataset) -> None:
    """
    Test inverse transform of TrendTransform.
    """
    original_df = example_tsds.to_pandas().copy(deep=True)
    trend_transform = TrendTransform(
        in_column="target",
        change_points_model=RupturesChangePointsModel(change_points_model=Binseg(model="rbf"), n_bkps=5),
        per_interval_model=SklearnRegressionPerIntervalModel(),
    )
    trend_transform.fit_transform(example_tsds)
    trend_transform.inverse_transform(example_tsds)
    pd.testing.assert_frame_equal(original_df, example_tsds[:, :, "target"])


def test_transform_interface_out_column(example_tsds: TSDataset) -> None:
    """Test transform interface with out_column param"""
    out_column = "regressor_test"
    trend_transform = TrendTransform(
        in_column="target",
        per_interval_model=SklearnRegressionPerIntervalModel(),
        change_points_model=RupturesChangePointsModel(change_points_model=Binseg(model="rbf"), n_bkps=5),
        out_column=out_column,
    )
    result = trend_transform.fit_transform(example_tsds).to_pandas()
    for seg in result.columns.get_level_values(0).unique():
        assert out_column in result[seg].columns


def test_transform_interface_repr(example_tsds: TSDataset) -> None:
    """Test transform interface without out_column param"""
    trend_transform = TrendTransform(
        in_column="target",
        per_interval_model=SklearnRegressionPerIntervalModel(),
        change_points_model=RupturesChangePointsModel(change_points_model=Binseg(model="rbf"), n_bkps=5),
    )
    out_column = f"{trend_transform.__repr__()}"
    result = trend_transform.fit_transform(example_tsds).to_pandas()
    for seg in result.columns.get_level_values(0).unique():
        assert out_column in result[seg].columns


@pytest.mark.parametrize("model", (LinearRegression, RandomForestRegressor))
def test_fit_transform_with_nans_in_tails(ts_with_nans_in_tails, model):
    transform = TrendTransform(
        in_column="target",
        per_interval_model=SklearnRegressionPerIntervalModel(model=model()),
        change_points_model=RupturesChangePointsModel(change_points_model=Binseg(model="rbf", jump=1), n_bkps=5),
        out_column="regressor_result",
    )
    transformed = transform.fit_transform(ts_with_nans_in_tails).to_pandas()
    for segment in transformed.columns.get_level_values("segment").unique():
        segment_slice = transformed.loc[pd.IndexSlice[:], pd.IndexSlice[segment, :]][segment]
        residue = segment_slice["target"] - segment_slice["regressor_result"]
        assert residue.mean() < 0.13


@pytest.mark.parametrize("model", (LinearRegression, RandomForestRegressor))
def test_fit_transform_with_nans_in_middle_raise_error(ts_with_nans, model):
    transform = TrendTransform(
        in_column="target",
        per_interval_model=SklearnRegressionPerIntervalModel(model=model()),
        change_points_model=RupturesChangePointsModel(change_points_model=Binseg(model="rbf"), n_bkps=5),
    )
    with pytest.raises(ValueError, match="The input column contains NaNs in the middle of the series!"):
        transform.fit_transform(ts_with_nans)


def test_save_load(example_tsds):
    transform = TrendTransform(
        in_column="target",
        per_interval_model=SklearnRegressionPerIntervalModel(),
        change_points_model=RupturesChangePointsModel(change_points_model=Binseg(model="rbf"), n_bkps=5),
    )
    assert_transformation_equals_loaded_original(transform=transform, ts=example_tsds)


@pytest.mark.parametrize(
    "transform, expected_length",
    [
        (TrendTransform(in_column="target"), 2),
        (
            TrendTransform(
                in_column="target",
                change_points_model=RupturesChangePointsModel(
                    change_points_model=Binseg(model="ar"),
                    n_bkps=5,
                ),
            ),
            2,
        ),
        (
            TrendTransform(
                in_column="target",
                change_points_model=RupturesChangePointsModel(
                    change_points_model=Binseg(model="ar"),
                    n_bkps=10,
                ),
            ),
            0,
        ),
    ],
)
def test_params_to_tune(transform, expected_length, example_tsds):
    ts = example_tsds
    assert len(transform.params_to_tune()) == expected_length
    assert_sampling_is_valid(transform=transform, ts=ts)
