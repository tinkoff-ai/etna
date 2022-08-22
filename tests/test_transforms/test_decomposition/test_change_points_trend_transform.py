import numpy as np
import pandas as pd
import pytest
from ruptures import Binseg
from sklearn.linear_model import LinearRegression

from etna.datasets import TSDataset
from etna.transforms.decomposition.change_points_trend import ChangePointsTrendTransform
from etna.transforms.decomposition.change_points_trend import _OneSegmentChangePointsTrendTransform


@pytest.fixture
def post_multitrend_df() -> pd.DataFrame:
    """Generate pd.DataFrame with timestamp after multitrend_df."""
    df = pd.DataFrame({"timestamp": pd.date_range("2021-07-01", "2021-07-31")})
    df["target"] = 0
    df["segment"] = "segment_1"
    df = TSDataset.to_dataset(df=df)
    return df


@pytest.fixture
def pre_multitrend_df() -> pd.DataFrame:
    """Generate pd.DataFrame with timestamp before multitrend_df."""
    df = pd.DataFrame({"timestamp": pd.date_range("2019-12-01", "2019-12-31")})
    df["target"] = 0
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


def test_models_after_fit(multitrend_df: pd.DataFrame):
    """Check that fit method generates correct number of detrend model's copies."""
    bs = _OneSegmentChangePointsTrendTransform(
        in_column="target", change_point_model=Binseg(), detrend_model=LinearRegression(), n_bkps=5
    )
    bs.fit(df=multitrend_df["segment_1"])
    assert isinstance(bs.per_interval_models, dict)
    assert len(bs.per_interval_models) == 6
    models = bs.per_interval_models.values()
    models_ids = [id(model) for model in models]
    assert len(set(models_ids)) == 6


def test_transform_detrend(multitrend_df: pd.DataFrame):
    """Check that transform method detrends given series."""
    bs = _OneSegmentChangePointsTrendTransform(
        in_column="target", change_point_model=Binseg(), detrend_model=LinearRegression(), n_bkps=5
    )
    bs.fit(df=multitrend_df["segment_1"])
    transformed = bs.transform(df=multitrend_df["segment_1"])
    assert transformed.columns == ["target"]
    assert abs(transformed["target"].mean()) < 0.1


def test_transform(multitrend_df: pd.DataFrame):
    """Check that detrend models get series trends."""
    bs = _OneSegmentChangePointsTrendTransform(
        in_column="target", change_point_model=Binseg(), detrend_model=LinearRegression(), n_bkps=50
    )
    bs.fit(df=multitrend_df["segment_1"])
    transformed = bs.transform(df=multitrend_df["segment_1"])
    assert transformed.columns == ["target"]
    assert abs(transformed["target"].std()) < 1


def test_inverse_transform(multitrend_df: pd.DataFrame):
    """Check that inverse_transform turns transformed series back to the origin one."""
    bs = _OneSegmentChangePointsTrendTransform(
        in_column="target", change_point_model=Binseg(), detrend_model=LinearRegression(), n_bkps=5
    )
    bs.fit(df=multitrend_df["segment_1"])

    transformed = bs.transform(df=multitrend_df["segment_1"].copy(deep=True))
    transformed_df_old = transformed.reset_index()
    transformed_df_old["segment"] = "segment_1"
    transformed_df = TSDataset.to_dataset(df=transformed_df_old)

    inversed = bs.inverse_transform(df=transformed_df["segment_1"].copy(deep=True))

    np.testing.assert_array_almost_equal(inversed["target"], multitrend_df["segment_1"]["target"], decimal=10)


def test_inverse_transform_hard(multitrend_df: pd.DataFrame):
    """Check the logic of out-of-sample inverse transformation: for past and future dates unseen by transform."""
    bs = _OneSegmentChangePointsTrendTransform(
        in_column="target", change_point_model=Binseg(), detrend_model=LinearRegression(), n_bkps=5
    )
    bs.fit(df=multitrend_df["segment_1"]["2020-02-01":"2021-05-01"])

    transformed = bs.transform(df=multitrend_df["segment_1"].copy(deep=True))
    transformed_df_old = transformed.reset_index()
    transformed_df_old["segment"] = "segment_1"
    transformed_df = TSDataset.to_dataset(df=transformed_df_old)

    inversed = bs.inverse_transform(df=transformed_df["segment_1"].copy(deep=True))

    np.testing.assert_array_almost_equal(inversed["target"], multitrend_df["segment_1"]["target"], decimal=10)


def test_transform_pre_history(multitrend_df: pd.DataFrame, pre_multitrend_df: pd.DataFrame):
    """Check that transform works correctly in case of fully unseen pre history data."""
    bs = _OneSegmentChangePointsTrendTransform(
        in_column="target", change_point_model=Binseg(), detrend_model=LinearRegression(), n_bkps=20
    )
    bs.fit(df=multitrend_df["segment_1"])
    transformed = bs.transform(pre_multitrend_df["segment_1"])
    expected = [x * 0.4 for x in list(range(31, 0, -1))]
    np.testing.assert_array_almost_equal(transformed["target"], expected, decimal=10)


def test_inverse_transform_pre_history(multitrend_df: pd.DataFrame, pre_multitrend_df: pd.DataFrame):
    """Check that inverse_transform works correctly in case of fully unseen pre history data."""
    bs = _OneSegmentChangePointsTrendTransform(
        in_column="target", change_point_model=Binseg(), detrend_model=LinearRegression(), n_bkps=20
    )
    bs.fit(df=multitrend_df["segment_1"])
    inversed = bs.inverse_transform(pre_multitrend_df["segment_1"])
    expected = [x * (-0.4) for x in list(range(31, 0, -1))]
    np.testing.assert_array_almost_equal(inversed["target"], expected, decimal=10)


def test_transform_post_history(multitrend_df: pd.DataFrame, post_multitrend_df: pd.DataFrame):
    """Check that transform works correctly in case of fully unseen post history data with offset."""
    bs = _OneSegmentChangePointsTrendTransform(
        in_column="target", change_point_model=Binseg(), detrend_model=LinearRegression(), n_bkps=20
    )
    bs.fit(df=multitrend_df["segment_1"])
    transformed = bs.transform(post_multitrend_df["segment_1"])
    # trend + last point of seen data + trend for offset interval
    expected = [abs(x * (-0.6) - 52.6 - 0.6 * 30) for x in list(range(1, 32))]
    np.testing.assert_array_almost_equal(transformed["target"], expected, decimal=10)


def test_inverse_transform_post_history(multitrend_df: pd.DataFrame, post_multitrend_df: pd.DataFrame):
    """Check that inverse_transform works correctly in case of fully unseen post history data with offset."""
    bs = _OneSegmentChangePointsTrendTransform(
        in_column="target", change_point_model=Binseg(), detrend_model=LinearRegression(), n_bkps=20
    )
    bs.fit(df=multitrend_df["segment_1"])
    transformed = bs.inverse_transform(post_multitrend_df["segment_1"])
    # trend + last point of seen data + trend for offset interval
    expected = [x * (-0.6) - 52.6 - 0.6 * 30 for x in list(range(1, 32))]
    np.testing.assert_array_almost_equal(transformed["target"], expected, decimal=10)


def test_transform_raise_error_if_not_fitted(multitrend_df: pd.DataFrame):
    """Test that transform for one segment raise error when calling transform without being fit."""
    transform = _OneSegmentChangePointsTrendTransform(
        in_column="target", change_point_model=Binseg(), detrend_model=LinearRegression(), n_bkps=5
    )
    with pytest.raises(ValueError, match="Transform is not fitted!"):
        _ = transform.transform(df=multitrend_df["segment_1"])


def test_fit_transform_with_nans_in_tails(multitrend_df_with_nans_in_tails):
    transform = ChangePointsTrendTransform(
        in_column="target", change_point_model=Binseg(), detrend_model=LinearRegression(), n_bkps=5
    )
    transformed = transform.fit_transform(df=multitrend_df_with_nans_in_tails)
    for segment in transformed.columns.get_level_values("segment").unique():
        segment_slice = transformed.loc[pd.IndexSlice[:], pd.IndexSlice[segment, :]][segment]
        assert abs(segment_slice["target"].mean()) < 0.1


def test_fit_transform_with_nans_in_middle_raise_error(df_with_nans):
    bs = ChangePointsTrendTransform(
        in_column="target", change_point_model=Binseg(), detrend_model=LinearRegression(), n_bkps=5
    )
    with pytest.raises(ValueError, match="The input column contains NaNs in the middle of the series!"):
        _ = bs.fit_transform(df=df_with_nans)
