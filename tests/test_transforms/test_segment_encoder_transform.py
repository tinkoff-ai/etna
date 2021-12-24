import numpy as np
import pandas as pd
import pytest

from etna.datasets import TSDataset
from etna.metrics import R2
from etna.models import LinearMultiSegmentModel
from etna.transforms import MeanSegmentEncoderTransform
from etna.transforms import SegmentEncoderTransform


@pytest.fixture
def dummy_df() -> pd.DataFrame:
    df_1 = pd.DataFrame.from_dict({"timestamp": pd.date_range("2021-06-01", "2021-07-01", freq="1d")})
    df_2 = pd.DataFrame.from_dict({"timestamp": pd.date_range("2021-06-01", "2021-07-01", freq="1d")})
    df_1["segment"] = "Moscow"
    df_1["target"] = 1
    df_2["segment"] = "Omsk"
    df_2["target"] = 2
    classic_df = pd.concat([df_1, df_2], ignore_index=True)

    df = classic_df.pivot(index="timestamp", columns="segment")
    df = df.reorder_levels([1, 0], axis=1)
    df = df.sort_index(axis=1)
    df.columns.names = ["segment", "feature"]
    return df


def test_segment_encoder_transform(dummy_df):
    transform = SegmentEncoderTransform()
    transformed_df = transform.fit_transform(dummy_df)
    assert (
        len(transformed_df.loc[:, pd.IndexSlice[:, "regressor_segment_code"]].columns) == 2
    ), "Number of columns not the same as segments"
    assert len(dummy_df) == len(transformed_df), "Row missing"
    codes = set()
    for segment in dummy_df.columns.get_level_values("segment").unique():
        column = transformed_df.loc[:, pd.IndexSlice[segment, "regressor_segment_code"]]
        assert column.dtype == "category", "Column type is not category"
        assert np.all(column == column.iloc[0]), "Values are not the same for the whole column"
        codes.add(column.iloc[0])
    assert codes == {0, 1}, "Codes are not 0 and 1"


@pytest.fixture
def simple_df() -> pd.DataFrame:
    df_1 = pd.DataFrame.from_dict({"timestamp": pd.date_range("2021-06-01", "2021-06-07", freq="D")})
    df_2 = pd.DataFrame.from_dict({"timestamp": pd.date_range("2021-06-01", "2021-06-07", freq="D")})
    df_1["segment"] = "Moscow"
    df_1["target"] = [1.0, 2.0, 3.0, 4.0, 5.0, np.NAN, np.NAN]
    df_1["exog"] = [6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    df_2["segment"] = "Omsk"
    df_2["target"] = [10.0, 20.0, 30.0, 40.0, 50.0, np.NAN, np.NAN]
    df_2["exog"] = [60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0]
    classic_df = pd.concat([df_1, df_2], ignore_index=True)
    df = TSDataset.to_dataset(classic_df)
    return df


@pytest.fixture
def transformed_simple_df() -> pd.DataFrame:
    df_1 = pd.DataFrame.from_dict({"timestamp": pd.date_range("2021-06-01", "2021-06-07", freq="D")})
    df_2 = pd.DataFrame.from_dict({"timestamp": pd.date_range("2021-06-01", "2021-06-07", freq="D")})
    df_1["segment"] = "Moscow"
    df_1["target"] = [1.0, 2.0, 3.0, 4.0, 5.0, np.NAN, np.NAN]
    df_1["exog"] = [6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    df_1["regressor_segment_mean"] = [0, 1, 1.5, 2, 2.5, 3, 3]
    df_2["segment"] = "Omsk"
    df_2["target"] = [10.0, 20.0, 30.0, 40.0, 50.0, np.NAN, np.NAN]
    df_2["exog"] = [60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0]
    df_2["regressor_segment_mean"] = [0.0, 10.0, 15.0, 20.0, 25.0, 30, 30]
    classic_df = pd.concat([df_1, df_2], ignore_index=True)
    df = TSDataset.to_dataset(classic_df)
    return df


@pytest.mark.parametrize("expected_global_means", ([[3, 30]]))
def test_mean_segment_encoder_fit(simple_df, expected_global_means):
    encoder = MeanSegmentEncoderTransform()
    encoder.fit(simple_df)
    assert (encoder.global_means == expected_global_means).all()


def test_mean_segment_encoder_transform(simple_df, transformed_simple_df):
    encoder = MeanSegmentEncoderTransform()
    transformed_df = encoder.fit_transform(simple_df)
    pd.testing.assert_frame_equal(transformed_df, transformed_simple_df)


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
