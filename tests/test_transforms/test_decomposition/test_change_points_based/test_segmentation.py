import numpy as np
import pandas as pd
import pytest
from ruptures import Binseg

from etna.datasets import TSDataset
from etna.datasets import generate_ar_df
from etna.metrics import SMAPE
from etna.models import CatBoostPerSegmentModel
from etna.pipeline import Pipeline
from etna.transforms import ChangePointsSegmentationTransform
from etna.transforms.decomposition.change_points_based.change_points_models import RupturesChangePointsModel
from etna.transforms.decomposition.change_points_based.segmentation import _OneSegmentChangePointsSegmentationTransform
from tests.test_transforms.utils import assert_sampling_is_valid
from tests.test_transforms.utils import assert_transformation_equals_loaded_original

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
    change_points_model = RupturesChangePointsModel(change_points_model=Binseg(), n_bkps=N_BKPS)
    bs = _OneSegmentChangePointsSegmentationTransform(
        in_column="target", change_points_model=change_points_model, out_column=OUT_COLUMN
    )
    bs.fit(df=pre_transformed_df["segment_1"])
    assert bs.intervals is not None


def test_transform_format_one_segment(pre_transformed_df: pd.DataFrame):
    """Check that transform method generate new column."""
    change_points_model = RupturesChangePointsModel(change_points_model=Binseg(), n_bkps=N_BKPS)
    bs = _OneSegmentChangePointsSegmentationTransform(
        in_column="target", change_points_model=change_points_model, out_column=OUT_COLUMN
    )
    bs.fit(df=pre_transformed_df["segment_1"])
    transformed = bs.transform(df=pre_transformed_df["segment_1"])
    assert set(transformed.columns) == {"target", OUT_COLUMN}
    assert transformed[OUT_COLUMN].dtype == "category"


def test_monotonously_result(pre_transformed_df: pd.DataFrame):
    """Check that resulting column is monotonously non-decreasing."""
    change_points_model = RupturesChangePointsModel(change_points_model=Binseg(), n_bkps=N_BKPS)
    bs = _OneSegmentChangePointsSegmentationTransform(
        in_column="target", change_points_model=change_points_model, out_column=OUT_COLUMN
    )
    bs.fit(df=pre_transformed_df["segment_1"])

    transformed = bs.transform(df=pre_transformed_df["segment_1"].copy(deep=True))
    result = transformed[OUT_COLUMN].astype(int).values
    assert (result[1:] - result[:-1] >= 0).mean() == 1


def test_transform_raise_error_if_not_fitted(pre_transformed_df: pd.DataFrame):
    """Test that transform for one segment raise error when calling transform without being fit."""
    change_points_model = RupturesChangePointsModel(change_points_model=Binseg(), n_bkps=N_BKPS)
    transform = _OneSegmentChangePointsSegmentationTransform(
        in_column="target", change_points_model=change_points_model, out_column=OUT_COLUMN
    )
    with pytest.raises(ValueError, match="Transform is not fitted!"):
        _ = transform.transform(df=pre_transformed_df["segment_1"])


def test_backtest(simple_ar_ts):
    model = CatBoostPerSegmentModel()
    horizon = 3
    change_points_model = RupturesChangePointsModel(change_points_model=Binseg(), n_bkps=N_BKPS)
    bs = ChangePointsSegmentationTransform(
        in_column="target", change_points_model=change_points_model, out_column=OUT_COLUMN
    )
    pipeline = Pipeline(model=model, transforms=[bs], horizon=horizon)
    _, _, _ = pipeline.backtest(ts=simple_ar_ts, metrics=[SMAPE()], n_folds=3)


def test_future_and_past_filling(simple_ar_ts):
    change_points_model = RupturesChangePointsModel(change_points_model=Binseg(), n_bkps=N_BKPS)
    bs = ChangePointsSegmentationTransform(
        in_column="target", change_points_model=change_points_model, out_column=OUT_COLUMN
    )
    before, ts = simple_ar_ts.train_test_split(test_start="2021-06-01")
    train, after = ts.train_test_split(test_start="2021-08-01")
    bs.fit_transform(ts=train)
    bs.transform(ts=before)
    bs.transform(ts=after)
    for seg in train.segments:
        assert np.sum(np.abs(before.to_pandas()[seg][OUT_COLUMN].astype(int))) == 0
        assert (after.to_pandas()[seg][OUT_COLUMN].astype(int) == 5).all()


def test_make_future(simple_ar_ts):
    change_points_model = RupturesChangePointsModel(change_points_model=Binseg(), n_bkps=N_BKPS)
    bs = ChangePointsSegmentationTransform(
        in_column="target", change_points_model=change_points_model, out_column=OUT_COLUMN
    )
    simple_ar_ts.fit_transform(transforms=[bs])
    future = simple_ar_ts.make_future(10, transforms=[bs])
    for seg in simple_ar_ts.segments:
        assert (future.to_pandas()[seg][OUT_COLUMN].astype(int) == 5).all()


def test_save_load(simple_ar_ts):
    change_points_model = RupturesChangePointsModel(change_points_model=Binseg(), n_bkps=N_BKPS)
    transform = ChangePointsSegmentationTransform(
        in_column="target", change_points_model=change_points_model, out_column=OUT_COLUMN
    )
    assert_transformation_equals_loaded_original(transform=transform, ts=simple_ar_ts)


@pytest.mark.parametrize(
    "transform, expected_length",
    [
        (ChangePointsSegmentationTransform(in_column="target"), 2),
        (
            ChangePointsSegmentationTransform(
                in_column="target",
                change_points_model=RupturesChangePointsModel(
                    change_points_model=Binseg(model="ar"),
                    n_bkps=5,
                ),
            ),
            2,
        ),
        (
            ChangePointsSegmentationTransform(
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
def test_params_to_tune(transform, expected_length, simple_ar_ts):
    ts = simple_ar_ts
    assert len(transform.params_to_tune()) == expected_length
    assert_sampling_is_valid(transform=transform, ts=ts)
