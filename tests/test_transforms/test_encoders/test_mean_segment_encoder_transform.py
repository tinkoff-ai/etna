import numpy as np
import pandas as pd
import pytest

from etna.datasets import TSDataset
from etna.metrics import R2
from etna.models import LinearMultiSegmentModel
from etna.transforms import MeanSegmentEncoderTransform
from tests.test_transforms.utils import assert_transformation_equals_loaded_original


@pytest.mark.parametrize("expected_global_means", [{"Moscow": 3, "Omsk": 30}])
def test_mean_segment_encoder_fit(simple_df, expected_global_means):
    encoder = MeanSegmentEncoderTransform()
    encoder.fit(simple_df)
    assert encoder.global_means == expected_global_means


def test_mean_segment_encoder_transform(simple_df, transformed_simple_df):
    encoder = MeanSegmentEncoderTransform()
    transformed_df = encoder.fit_transform(simple_df)
    pd.testing.assert_frame_equal(transformed_df, transformed_simple_df)


def test_subset_segments(simple_df):
    train_df = simple_df
    test_df = simple_df.loc[:, pd.IndexSlice["Omsk", :]]
    transform = MeanSegmentEncoderTransform()

    transform.fit(train_df)
    transformed_test_df = transform.transform(test_df)

    segments = sorted(transformed_test_df.columns.get_level_values("segment").unique())
    features = sorted(transformed_test_df.columns.get_level_values("feature").unique())
    assert segments == ["Omsk"]
    assert features == ["exog", "segment_mean", "target"]


def test_not_fitted_error(simple_df):
    encoder = MeanSegmentEncoderTransform()
    with pytest.raises(ValueError, match="The transform isn't fitted"):
        encoder.transform(simple_df)


def test_new_segments_error(simple_df):
    train_df = simple_df.loc[:, pd.IndexSlice["Moscow", :]]
    test_df = simple_df.loc[:, pd.IndexSlice["Omsk", :]]
    transform = MeanSegmentEncoderTransform()

    transform.fit(train_df)
    with pytest.raises(
        NotImplementedError, match="This transform can't process segments that weren't present on train data"
    ):
        _ = transform.transform(test_df)


@pytest.fixture
def almost_constant_ts(random_seed) -> TSDataset:
    df_1 = pd.DataFrame.from_dict({"timestamp": pd.date_range("2021-06-01", "2021-07-01", freq="D")})
    df_2 = pd.DataFrame.from_dict({"timestamp": pd.date_range("2021-06-01", "2021-07-01", freq="D")})
    df_1["segment"] = "Moscow"
    df_1["target"] = 1 + np.random.normal(0, 0.1, size=len(df_1))
    df_2["segment"] = "Omsk"
    df_2["target"] = 10 + np.random.normal(0, 0.1, size=len(df_1))
    classic_df = pd.concat([df_1, df_2], ignore_index=True)
    ts = TSDataset(df=TSDataset.to_dataset(classic_df), freq="D")
    return ts


def test_mean_segment_encoder_forecast(almost_constant_ts):
    """Test that MeanSegmentEncoderTransform works correctly in forecast pipeline
    and helps to correctly forecast almost constant series."""
    horizon = 5
    model = LinearMultiSegmentModel()
    encoder = MeanSegmentEncoderTransform()

    train, test = almost_constant_ts.train_test_split(test_size=horizon)
    train.fit_transform([encoder])
    model.fit(train)
    future = train.make_future(horizon)
    pred_mean_segment_encoding = model.forecast(future)

    metric = R2(mode="macro")

    # R2=0 => model predicts the optimal constant
    assert np.allclose(metric(pred_mean_segment_encoding, test), 0)


def test_fit_transform_with_nans(ts_diff_endings):
    encoder = MeanSegmentEncoderTransform()
    ts_diff_endings.fit_transform([encoder])


def test_save_load(almost_constant_ts):
    transform = MeanSegmentEncoderTransform()
    assert_transformation_equals_loaded_original(transform=transform, ts=almost_constant_ts)
