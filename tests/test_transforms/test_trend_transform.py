from copy import deepcopy

import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from etna.datasets.tsdataset import TSDataset
from ruptures import Binseg
from etna.transforms.trend import TrendTransform
from etna.transforms.trend import _OneSegmentTrendTransform
from etna.transforms.trend import _TrendTransform

DEFAULT_SEGMENT = "segment_1"


@pytest.fixture
def df_one_segment(example_df) -> pd.DataFrame:
    return example_df[example_df["segment"] == DEFAULT_SEGMENT].set_index("timestamp")


def test_fit_transform_one_segment(df_one_segment: pd.DataFrame) -> None:
    """
    Test that fit_transform interface works correctly for one segment.
    """
    df_one_segment_original = df_one_segment.copy()
    trend_transform = _OneSegmentTrendTransform(
        in_column="target", change_point_model=Binseg(), detrend_model=LinearRegression(), n_bkps=5
    )
    df_one_segment = trend_transform.fit_transform(df_one_segment)
    assert sorted(df_one_segment.columns) == sorted(["target", "segment", "regressor_target_trend"])
    assert (df_one_segment["target"] == df_one_segment_original["target"]).all()
    residue = df_one_segment["target"] - df_one_segment["regressor_target_trend"]
    assert residue.mean() < 1


def test_inverse_transform_one_segment(df_one_segment: pd.DataFrame) -> None:
    """
    Test that inverse_transform interface works correctly for one segment.
    """
    trend_transform = _OneSegmentTrendTransform(
        in_column="target", change_point_model=Binseg(), detrend_model=LinearRegression(), n_bkps=5
    )
    df_one_segment_transformed = trend_transform.fit_transform(df_one_segment)
    df_one_segment_inverse_transformed = trend_transform.inverse_transform(df_one_segment)
    assert (df_one_segment_transformed == df_one_segment_inverse_transformed).all().all()


def test_fit_transform_many_segments(example_tsds: TSDataset) -> None:
    """
    Test that fit_transform interface works correctly for many segment.
    """
    example_tsds_original = deepcopy(example_tsds)
    trend_transform = _TrendTransform(
        in_column="target", change_point_model=Binseg(), detrend_model=LinearRegression(), n_bkps=5
    )
    example_tsds.fit_transform([trend_transform])
    for segment in example_tsds.segments:
        segment_slice = example_tsds[:, segment, :][segment]
        segment_slice_original = example_tsds_original[:, segment, :][segment]
        assert sorted(segment_slice.columns) == sorted(["target", "regressor_target_trend"])
        assert (segment_slice["target"] == segment_slice_original["target"]).all()
        residue = segment_slice_original["target"] - segment_slice["regressor_target_trend"]
        assert residue.mean() < 1


def test_inverse_transform_many_segments(example_tsds: TSDataset) -> None:
    """
    Test that inverse_transform interface works correctly for many segment.
    """
    trend_transform = _TrendTransform(
        in_column="target", change_point_model=Binseg(), detrend_model=LinearRegression(), n_bkps=5
    )
    example_tsds_transformed = example_tsds.fit_transform([trend_transform])
    example_tsds_inverse_transformed = example_tsds.inverse_transform()
    assert example_tsds_transformed == example_tsds_inverse_transformed


def test_transform_run(example_tsds: TSDataset) -> None:
    """
    Test interface of TrendTransform.
    """
    trend_transform = TrendTransform(in_column="target", detrend_model=LinearRegression(), model="rbf")
    example_tsds_transformed = example_tsds.fit_transform([trend_transform])
    example_tsds_inverse_transformed = example_tsds.inverse_transform()
    assert example_tsds_transformed == example_tsds_inverse_transformed
