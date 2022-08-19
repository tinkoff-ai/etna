import numpy as np
import pandas as pd
import pytest
from ruptures import Binseg

from etna.datasets import TSDataset
from etna.datasets import generate_ar_df
from etna.metrics import SMAPE
from etna.models import CatBoostModelPerSegment
from etna.pipeline import Pipeline
from etna.transforms import ChangePointsSegmentationTransform
from etna.transforms.decomposition.base_change_points import RupturesChangePointsModel
from etna.transforms.decomposition.change_points_segmentation import _OneSegmentChangePointsSegmentationTransform

OUT_COLUMN = "result"
N_BKPS = 5


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
def multitrend_df_with_nans_in_tails(multitrend_df):
    multitrend_df.loc[
        [multitrend_df.index[0], multitrend_df.index[1], multitrend_df.index[-2], multitrend_df.index[-1]],
        pd.IndexSlice["segment_1", "target"],
    ] = None
    return multitrend_df


def test_fit_one_segment(pre_transformed_df: pd.DataFrame):
    """Check that fit method save intervals."""
    change_point_model = RupturesChangePointsModel(change_point_model=Binseg(), n_bkps=N_BKPS)
    bs = _OneSegmentChangePointsSegmentationTransform(
        in_column="target", change_point_model=change_point_model, out_column=OUT_COLUMN
    )
    bs.fit(df=pre_transformed_df["segment_1"])
    assert bs.intervals is not None


def test_transform_format_one_segment(pre_transformed_df: pd.DataFrame):
    """Check that transform method generate new column."""
    change_point_model = RupturesChangePointsModel(change_point_model=Binseg(), n_bkps=N_BKPS)
    bs = _OneSegmentChangePointsSegmentationTransform(
        in_column="target", change_point_model=change_point_model, out_column=OUT_COLUMN
    )
    bs.fit(df=pre_transformed_df["segment_1"])
    transformed = bs.transform(df=pre_transformed_df["segment_1"])
    assert set(transformed.columns) == {"target", OUT_COLUMN}
    assert transformed[OUT_COLUMN].dtype == "category"


def test_monotonously_result(pre_transformed_df: pd.DataFrame):
    """Check that resulting column is monotonously non-decreasing."""
    change_point_model = RupturesChangePointsModel(change_point_model=Binseg(), n_bkps=N_BKPS)
    bs = _OneSegmentChangePointsSegmentationTransform(
        in_column="target", change_point_model=change_point_model, out_column=OUT_COLUMN
    )
    bs.fit(df=pre_transformed_df["segment_1"])

    transformed = bs.transform(df=pre_transformed_df["segment_1"].copy(deep=True))
    result = transformed[OUT_COLUMN].astype(int).values
    assert (result[1:] - result[:-1] >= 0).mean() == 1


def test_transform_raise_error_if_not_fitted(pre_transformed_df: pd.DataFrame):
    """Test that transform for one segment raise error when calling transform without being fit."""
    change_point_model = RupturesChangePointsModel(change_point_model=Binseg(), n_bkps=N_BKPS)
    transform = _OneSegmentChangePointsSegmentationTransform(
        in_column="target", change_point_model=change_point_model, out_column=OUT_COLUMN
    )
    with pytest.raises(ValueError, match="Transform is not fitted!"):
        _ = transform.transform(df=pre_transformed_df["segment_1"])


def test_backtest(simple_ar_ts):
    model = CatBoostModelPerSegment()
    horizon = 3
    change_point_model = RupturesChangePointsModel(change_point_model=Binseg(), n_bkps=N_BKPS)
    bs = ChangePointsSegmentationTransform(
        in_column="target", change_point_model=change_point_model, out_column=OUT_COLUMN
    )
    pipeline = Pipeline(model=model, transforms=[bs], horizon=horizon)
    _, _, _ = pipeline.backtest(ts=simple_ar_ts, metrics=[SMAPE()], n_folds=3)


def test_future_and_past_filling(simple_ar_ts):
    change_point_model = RupturesChangePointsModel(change_point_model=Binseg(), n_bkps=N_BKPS)
    bs = ChangePointsSegmentationTransform(
        in_column="target", change_point_model=change_point_model, out_column=OUT_COLUMN
    )
    before, ts = simple_ar_ts.train_test_split(test_start="2021-06-01")
    train, after = ts.train_test_split(test_start="2021-08-01")
    bs.fit_transform(train.df)
    before = bs.transform(before.df)
    after = bs.transform(after.df)
    for seg in train.segments:
        assert np.sum(np.abs(before[seg][OUT_COLUMN].astype(int))) == 0
        assert (after[seg][OUT_COLUMN].astype(int) == 5).all()


def test_make_future(simple_ar_ts):
    change_point_model = RupturesChangePointsModel(change_point_model=Binseg(), n_bkps=N_BKPS)
    bs = ChangePointsSegmentationTransform(
        in_column="target", change_point_model=change_point_model, out_column=OUT_COLUMN
    )
    simple_ar_ts.fit_transform(transforms=[bs])
    future = simple_ar_ts.make_future(10)
    for seg in simple_ar_ts.segments:
        assert (future.to_pandas()[seg][OUT_COLUMN].astype(int) == 5).all()
