import numpy as np
import pandas as pd
import pytest
from ruptures import Binseg

from etna.datasets import TSDataset
from etna.datasets import generate_ar_df
from etna.metrics import SMAPE
from etna.models import CatBoostModelPerSegment
from etna.pipeline import Pipeline
from etna.transforms import ChangePointSegmentationTransform
from etna.transforms.encoders.change_point_segmentation import _OneSegmentChangePointSegmentationTransform


@pytest.fixture
def pre_transformed_df() -> pd.DataFrame:
    """Generate pd.DataFrame with timestamp."""
    df = pd.DataFrame({"timestamp": pd.date_range("2019-12-01", "2019-12-31")})
    df["target"] = 0
    df["segment"] = "segment_1"
    df = TSDataset.to_dataset(df=df)
    return df


@pytest.fixture
def simple_ar_ts(random_seed):
    df = generate_ar_df(periods=125, start_time="2021-05-20", n_segments=3, ar_coef=[2], freq="D")
    df_ts_format = TSDataset.to_dataset(df)
    return TSDataset(df_ts_format, freq="D")


@pytest.fixture
def df_with_nans() -> pd.DataFrame:
    """Generate pd.DataFrame with timestamp."""
    df = pd.DataFrame({"timestamp": pd.date_range("2019-12-01", "2019-12-31")})
    tmp = np.zeros(31)
    tmp[8] = None
    df["target"] = tmp
    df["segment"] = "segment_1"
    df = TSDataset.to_dataset(df=df)
    return df


@pytest.fixture
def multitrend_df_with_nans_in_tails(multitrend_df):
    multitrend_df.loc[
        [multitrend_df.index[0], multitrend_df.index[1], multitrend_df.index[-2], multitrend_df.index[-1]],
        pd.IndexSlice["segment_1", "target"],
    ] = None
    return multitrend_df


def test_build_intervals():
    """Check correctness of intervals generation with list of change points."""
    change_points = [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-18"), pd.Timestamp("2020-02-24")]
    expected_intervals = [
        (pd.Timestamp.min, pd.Timestamp("2020-01-01")),
        (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-18")),
        (pd.Timestamp("2020-01-18"), pd.Timestamp("2020-02-24")),
        (pd.Timestamp("2020-02-24"), pd.Timestamp.max),
    ]
    intervals = _OneSegmentChangePointSegmentationTransform._build_intervals(change_points=change_points)
    assert isinstance(intervals, list)
    assert len(intervals) == 4
    for (exp_left, exp_right), (real_left, real_right) in zip(expected_intervals, intervals):
        assert exp_left == real_left
        assert exp_right == real_right


def test_fit(pre_transformed_df: pd.DataFrame):
    """Check that fit method save intervals."""
    out_column = "result"
    bs = _OneSegmentChangePointSegmentationTransform(
        in_column="target", change_point_model=Binseg(), out_column=out_column, n_bkps=5
    )
    bs.fit(df=pre_transformed_df["segment_1"])
    assert bs.intervals is not None


def test_transform(pre_transformed_df: pd.DataFrame):
    """Check that transform method generate new column."""
    out_column = "result"
    bs = _OneSegmentChangePointSegmentationTransform(
        in_column="target", change_point_model=Binseg(), out_column=out_column, n_bkps=5
    )
    bs.fit(df=pre_transformed_df["segment_1"])
    transformed = bs.transform(df=pre_transformed_df["segment_1"])
    assert set(transformed.columns) == {"target", out_column}
    assert transformed[out_column].dtype == "category"


def test_inverse_transform(pre_transformed_df: pd.DataFrame):
    """Check that inverse_transform works."""
    out_column = "result"
    bs = _OneSegmentChangePointSegmentationTransform(
        in_column="target", change_point_model=Binseg(), out_column=out_column, n_bkps=5
    )
    bs.fit(df=pre_transformed_df["segment_1"])

    transformed = bs.transform(df=pre_transformed_df["segment_1"].copy(deep=True))
    bs.inverse_transform(transformed)


def test_monotonously_result(pre_transformed_df: pd.DataFrame):
    """Check that resulting column is monotonously non-decreasing."""
    out_column = "result"
    bs = _OneSegmentChangePointSegmentationTransform(
        in_column="target", change_point_model=Binseg(), out_column=out_column, n_bkps=5
    )
    bs.fit(df=pre_transformed_df["segment_1"])

    transformed = bs.transform(df=pre_transformed_df["segment_1"].copy(deep=True))
    result = transformed[out_column].astype(int).values
    assert (result[1:] - result[:-1] >= 0).mean() == 1


def test_transform_raise_error_if_not_fitted(pre_transformed_df: pd.DataFrame):
    """Test that transform for one segment raise error when calling transform without being fit."""
    out_column = "result"
    transform = _OneSegmentChangePointSegmentationTransform(
        in_column="target", change_point_model=Binseg(), out_column=out_column, n_bkps=5
    )
    with pytest.raises(ValueError, match="Transform is not fitted!"):
        _ = transform.transform(df=pre_transformed_df["segment_1"])


def test_fit_transform_with_nans_in_middle_raise_error(df_with_nans):
    out_column = "result"
    bs = _OneSegmentChangePointSegmentationTransform(
        in_column="target", change_point_model=Binseg(), out_column=out_column, n_bkps=5
    )
    with pytest.raises(ValueError, match="The input column contains NaNs in the middle of the series!"):
        _ = bs.fit_transform(df=df_with_nans["segment_1"])


def test_backtest(simple_ar_ts):
    model = CatBoostModelPerSegment()
    horizon = 3
    out_column = "result"
    bs = ChangePointSegmentationTransform(
        in_column="target", change_point_model=Binseg(), out_column=out_column, n_bkps=5
    )
    pipeline = Pipeline(model=model, transforms=[bs], horizon=horizon)
    _, _, _ = pipeline.backtest(ts=simple_ar_ts, metrics=[SMAPE()], n_folds=3)


def test_future_and_past_filling(simple_ar_ts):
    out_column = "result"
    bs = ChangePointSegmentationTransform(
        in_column="target", change_point_model=Binseg(), out_column=out_column, n_bkps=5
    )
    before, ts = simple_ar_ts.train_test_split(test_start="2021-06-01")
    train, after = ts.train_test_split(test_start="2021-08-01")
    bs.fit_transform(train.df)
    before = bs.transform(before.df)
    after = bs.transform(after.df)
    for seg in train.segments:
        assert np.sum(np.abs(before[seg][out_column].astype(int))) == 0
        assert np.sum(np.abs(after[seg][out_column].astype(int) - 5)) == 0
