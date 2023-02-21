from copy import deepcopy
from typing import Set
from typing import Tuple

import numpy as np
import pandas as pd
import pytest
from pandas.util.testing import assert_frame_equal
from ruptures import Binseg
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from etna.analysis import StatisticsRelevanceTable
from etna.datasets import TSDataset
from etna.models import ProphetModel
from etna.transforms import AddConstTransform
from etna.transforms import BinsegTrendTransform
from etna.transforms import BoxCoxTransform
from etna.transforms import ChangePointsSegmentationTransform
from etna.transforms import ChangePointsTrendTransform
from etna.transforms import DateFlagsTransform
from etna.transforms import DensityOutliersTransform
from etna.transforms import DifferencingTransform
from etna.transforms import FilterFeaturesTransform
from etna.transforms import FourierTransform
from etna.transforms import GaleShapleyFeatureSelectionTransform
from etna.transforms import HolidayTransform
from etna.transforms import LabelEncoderTransform
from etna.transforms import LagTransform
from etna.transforms import LambdaTransform
from etna.transforms import LinearTrendTransform
from etna.transforms import LogTransform
from etna.transforms import MADTransform
from etna.transforms import MaxAbsScalerTransform
from etna.transforms import MaxTransform
from etna.transforms import MeanSegmentEncoderTransform
from etna.transforms import MeanTransform
from etna.transforms import MedianOutliersTransform
from etna.transforms import MedianTransform
from etna.transforms import MinMaxDifferenceTransform
from etna.transforms import MinMaxScalerTransform
from etna.transforms import MinTransform
from etna.transforms import MRMRFeatureSelectionTransform
from etna.transforms import OneHotEncoderTransform
from etna.transforms import PredictionIntervalOutliersTransform
from etna.transforms import QuantileTransform
from etna.transforms import ResampleWithDistributionTransform
from etna.transforms import RobustScalerTransform
from etna.transforms import SegmentEncoderTransform
from etna.transforms import SpecialDaysTransform
from etna.transforms import StandardScalerTransform
from etna.transforms import StdTransform
from etna.transforms import STLTransform
from etna.transforms import SumTransform
from etna.transforms import TheilSenTrendTransform
from etna.transforms import TimeFlagsTransform
from etna.transforms import TimeSeriesImputerTransform
from etna.transforms import TreeFeatureSelectionTransform
from etna.transforms import TrendTransform
from etna.transforms import YeoJohnsonTransform
from etna.transforms.decomposition import RupturesChangePointsModel
from tests.utils import select_segments_subset
from tests.utils import to_be_fixed


@pytest.fixture
def regular_ts(random_seed) -> TSDataset:
    periods = 100
    df_1 = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=periods)})
    df_1["segment"] = "segment_1"
    df_1["target"] = np.random.uniform(10, 20, size=periods)

    df_2 = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=periods)})
    df_2["segment"] = "segment_2"
    df_2["target"] = np.random.uniform(-15, 5, size=periods)

    df_3 = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=periods)})
    df_3["segment"] = "segment_3"
    df_3["target"] = np.random.uniform(-5, 5, size=periods)

    df = pd.concat([df_1, df_2, df_3]).reset_index(drop=True)
    df = TSDataset.to_dataset(df)
    tsds = TSDataset(df, freq="D")

    return tsds


@pytest.fixture
def ts_with_exog(regular_ts) -> TSDataset:
    df = regular_ts.to_pandas(flatten=True)
    df_exog = df.copy().drop(columns=["target"])
    df_exog["weekday"] = df_exog["timestamp"].dt.weekday
    df_exog["monthday"] = df_exog["timestamp"].dt.day
    df_exog["month"] = df_exog["timestamp"].dt.month
    df_exog["year"] = df_exog["timestamp"].dt.year
    ts = TSDataset(df=TSDataset.to_dataset(df).iloc[5:-5], df_exog=TSDataset.to_dataset(df_exog), freq="D")
    return ts


@pytest.fixture
def positive_ts() -> TSDataset:
    periods = 100
    df_1 = pd.DataFrame.from_dict({"timestamp": pd.date_range("2020-01-01", periods=periods, freq="D")})
    df_2 = pd.DataFrame.from_dict({"timestamp": pd.date_range("2020-01-01", periods=periods, freq="D")})
    df_3 = pd.DataFrame.from_dict({"timestamp": pd.date_range("2020-01-01", periods=periods, freq="D")})
    generator = np.random.RandomState(seed=1)

    df_1["segment"] = "segment_1"
    df_1["target"] = np.abs(generator.normal(loc=10, scale=1, size=len(df_1))) + 1

    df_2["segment"] = "segment_2"
    df_2["target"] = np.abs(generator.normal(loc=20, scale=1, size=len(df_2))) + 1

    df_3["segment"] = "segment_3"
    df_3["target"] = np.abs(generator.normal(loc=30, scale=1, size=len(df_2))) + 1

    classic_df = pd.concat([df_1, df_2, df_3], ignore_index=True)
    wide_df = TSDataset.to_dataset(classic_df)
    ts = TSDataset(df=wide_df, freq="D")
    return ts


@pytest.fixture
def ts_to_fill(regular_ts) -> TSDataset:
    df = regular_ts.to_pandas()
    df.iloc[5, 0] = np.NaN
    df.iloc[10, 1] = np.NaN
    df.iloc[20, 2] = np.NaN
    df.iloc[-5, 0] = np.NaN
    df.iloc[-10, 1] = np.NaN
    df.iloc[-20, 2] = np.NaN
    ts = TSDataset(df=df, freq="D")
    return ts


@pytest.fixture
def ts_to_resample() -> TSDataset:
    df_1 = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2020-01-05", freq="H", periods=120),
            "segment": "segment_1",
            "target": 1,
        }
    )
    df_2 = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2020-01-05", freq="H", periods=120),
            "segment": "segment_2",
            "target": ([1] + 23 * [0]) * 5,
        }
    )
    df_3 = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2020-01-05", freq="H", periods=120),
            "segment": "segment_3",
            "target": ([4] + 23 * [0]) * 5,
        }
    )
    df = pd.concat([df_1, df_2, df_3], ignore_index=True)

    df_exog_1 = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2020-01-05", freq="D", periods=8),
            "segment": "segment_1",
            "regressor_exog": 2,
        }
    )
    df_exog_2 = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2020-01-05", freq="D", periods=8),
            "segment": "segment_2",
            "regressor_exog": 40,
        }
    )
    df_exog_3 = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2020-01-05", freq="D", periods=8),
            "segment": "segment_3",
            "regressor_exog": 40,
        }
    )
    df_exog = pd.concat([df_exog_1, df_exog_2, df_exog_3], ignore_index=True)
    ts = TSDataset(df=TSDataset.to_dataset(df), freq="H", df_exog=TSDataset.to_dataset(df_exog), known_future="all")
    return ts


@pytest.fixture
def ts_with_outliers(regular_ts) -> TSDataset:
    df = regular_ts.to_pandas()
    df.iloc[5, 0] *= 100
    df.iloc[10, 1] *= 100
    df.iloc[20, 2] *= 100
    df.iloc[-5, 0] *= 100
    df.iloc[-10, 1] *= 100
    df.iloc[-20, 2] *= 100
    ts = TSDataset(df=df, freq="D")
    return ts


class TestTransformTrainSubsetSegments:
    """Test transform on train part of subset of segments.

    Expected that transformation on subset of segments match subset of transformation on full dataset.
    """

    def _test_transform_train_subset_segments(self, ts, transform, segments):
        # select subset of tsdataset
        segments = list(set(segments))
        subset_ts = select_segments_subset(ts=deepcopy(ts), segments=segments)
        df = ts.to_pandas()
        subset_df = subset_ts.to_pandas()

        # fitting
        transform.fit(df)

        # transform full
        transformed_df = transform.transform(df)

        # transform subset of segments
        transformed_subset_df = transform.transform(subset_df)

        # checking
        assert_frame_equal(transformed_subset_df, transformed_df.loc[:, pd.IndexSlice[segments, :]])

    @pytest.mark.parametrize(
        "transform, dataset_name",
        [
            # decomposition
            (
                ChangePointsSegmentationTransform(
                    in_column="target",
                    change_point_model=RupturesChangePointsModel(change_point_model=Binseg(), n_bkps=5),
                ),
                "regular_ts",
            ),
            (
                ChangePointsTrendTransform(
                    in_column="target", change_point_model=Binseg(), detrend_model=LinearRegression(), n_bkps=5
                ),
                "regular_ts",
            ),
            (BinsegTrendTransform(in_column="target"), "regular_ts"),
            (LinearTrendTransform(in_column="target"), "regular_ts"),
            (TheilSenTrendTransform(in_column="target"), "regular_ts"),
            (STLTransform(in_column="target", period=7), "regular_ts"),
            (TrendTransform(in_column="target"), "regular_ts"),
            # encoders
            (LabelEncoderTransform(in_column="weekday"), "ts_with_exog"),
            (OneHotEncoderTransform(in_column="weekday"), "ts_with_exog"),
            (MeanSegmentEncoderTransform(), "regular_ts"),
            (SegmentEncoderTransform(), "regular_ts"),
            # feature_selection
            (FilterFeaturesTransform(exclude=["year"]), "ts_with_exog"),
            (GaleShapleyFeatureSelectionTransform(relevance_table=StatisticsRelevanceTable(), top_k=2), "ts_with_exog"),
            (MRMRFeatureSelectionTransform(relevance_table=StatisticsRelevanceTable(), top_k=2), "ts_with_exog"),
            (TreeFeatureSelectionTransform(model=DecisionTreeRegressor(random_state=42), top_k=2), "ts_with_exog"),
            # math
            (AddConstTransform(in_column="target", value=1, inplace=False), "regular_ts"),
            (AddConstTransform(in_column="target", value=1, inplace=True), "regular_ts"),
            (LagTransform(in_column="target", lags=[1, 2, 3]), "regular_ts"),
            (
                LambdaTransform(in_column="target", transform_func=lambda x: x + 1, inplace=False),
                "regular_ts",
            ),
            (
                LambdaTransform(
                    in_column="target",
                    transform_func=lambda x: x + 1,
                    inverse_transform_func=lambda x: x - 1,
                    inplace=True,
                ),
                "regular_ts",
            ),
            (LogTransform(in_column="target", inplace=False), "positive_ts"),
            (LogTransform(in_column="target", inplace=True), "positive_ts"),
            (DifferencingTransform(in_column="target", inplace=False), "regular_ts"),
            (DifferencingTransform(in_column="target", inplace=True), "regular_ts"),
            (MADTransform(in_column="target", window=7), "regular_ts"),
            (MaxTransform(in_column="target", window=7), "regular_ts"),
            (MeanTransform(in_column="target", window=7), "regular_ts"),
            (MedianTransform(in_column="target", window=7), "regular_ts"),
            (MinMaxDifferenceTransform(in_column="target", window=7), "regular_ts"),
            (MinTransform(in_column="target", window=7), "regular_ts"),
            (QuantileTransform(in_column="target", quantile=0.9, window=7), "regular_ts"),
            (StdTransform(in_column="target", window=7), "regular_ts"),
            (SumTransform(in_column="target", window=7), "regular_ts"),
            (BoxCoxTransform(in_column="target", mode="per-segment", inplace=False), "positive_ts"),
            (BoxCoxTransform(in_column="target", mode="per-segment", inplace=True), "positive_ts"),
            (BoxCoxTransform(in_column="target", mode="macro", inplace=False), "positive_ts"),
            (BoxCoxTransform(in_column="target", mode="macro", inplace=True), "positive_ts"),
            (MaxAbsScalerTransform(in_column="target", mode="per-segment", inplace=False), "regular_ts"),
            (MaxAbsScalerTransform(in_column="target", mode="per-segment", inplace=True), "regular_ts"),
            (MaxAbsScalerTransform(in_column="target", mode="macro", inplace=False), "regular_ts"),
            (MaxAbsScalerTransform(in_column="target", mode="macro", inplace=True), "regular_ts"),
            (MinMaxScalerTransform(in_column="target", mode="per-segment", inplace=False), "regular_ts"),
            (MinMaxScalerTransform(in_column="target", mode="per-segment", inplace=True), "regular_ts"),
            (MinMaxScalerTransform(in_column="target", mode="macro", inplace=False), "regular_ts"),
            (MinMaxScalerTransform(in_column="target", mode="macro", inplace=True), "regular_ts"),
            (RobustScalerTransform(in_column="target", mode="per-segment", inplace=False), "regular_ts"),
            (RobustScalerTransform(in_column="target", mode="per-segment", inplace=True), "regular_ts"),
            (RobustScalerTransform(in_column="target", mode="macro", inplace=False), "regular_ts"),
            (RobustScalerTransform(in_column="target", mode="macro", inplace=True), "regular_ts"),
            (StandardScalerTransform(in_column="target", mode="per-segment", inplace=False), "regular_ts"),
            (StandardScalerTransform(in_column="target", mode="per-segment", inplace=True), "regular_ts"),
            (StandardScalerTransform(in_column="target", mode="macro", inplace=False), "regular_ts"),
            (StandardScalerTransform(in_column="target", mode="macro", inplace=True), "regular_ts"),
            (YeoJohnsonTransform(in_column="target", mode="per-segment", inplace=False), "regular_ts"),
            (YeoJohnsonTransform(in_column="target", mode="per-segment", inplace=True), "regular_ts"),
            (YeoJohnsonTransform(in_column="target", mode="macro", inplace=False), "regular_ts"),
            (YeoJohnsonTransform(in_column="target", mode="macro", inplace=True), "regular_ts"),
            # missing_values
            (
                ResampleWithDistributionTransform(
                    in_column="regressor_exog",
                    distribution_column="target",
                    inplace=False,
                ),
                "ts_to_resample",
            ),
            (
                ResampleWithDistributionTransform(
                    in_column="regressor_exog", distribution_column="target", inplace=True
                ),
                "ts_to_resample",
            ),
            (TimeSeriesImputerTransform(in_column="target"), "ts_to_fill"),
            # outliers
            (DensityOutliersTransform(in_column="target"), "ts_with_outliers"),
            (MedianOutliersTransform(in_column="target"), "ts_with_outliers"),
            (PredictionIntervalOutliersTransform(in_column="target", model=ProphetModel), "ts_with_outliers"),
            # timestamp
            (DateFlagsTransform(), "regular_ts"),
            (FourierTransform(period=7, order=2), "regular_ts"),
            (HolidayTransform(), "regular_ts"),
            (SpecialDaysTransform(), "regular_ts"),
            (TimeFlagsTransform(), "regular_ts"),
        ],
    )
    def test_transform_train_subset_segments(self, transform, dataset_name, request):
        ts = request.getfixturevalue(dataset_name)
        self._test_transform_train_subset_segments(ts, transform, segments=["segment_2"])


class TestTransformFutureSubsetSegments:
    """Test transform on future part of subset of segments.

    Expected that transformation on subset of segments match subset of transformation on full dataset.
    """

    def _test_transform_future_subset_segments(self, ts, transform, segments, horizon=7):
        # select subset of tsdataset
        subset_ts = select_segments_subset(ts=deepcopy(ts), segments=segments)
        train_df = ts.to_pandas()
        ts.transforms = [transform]
        subset_ts.transforms = [transform]

        # fitting
        transform.fit(train_df)

        # transform full
        transformed_future_ts = ts.make_future(future_steps=horizon)

        # transform subset of segments
        transformed_subset_future_ts = subset_ts.make_future(future_steps=horizon)

        # checking
        transformed_future_df = transformed_future_ts.to_pandas()
        transformed_subset_future_df = transformed_subset_future_ts.to_pandas()
        assert_frame_equal(transformed_subset_future_df, transformed_future_df.loc[:, pd.IndexSlice[segments, :]])

    @pytest.mark.parametrize(
        "transform, dataset_name",
        [
            # decomposition
            (
                ChangePointsSegmentationTransform(
                    in_column="target",
                    change_point_model=RupturesChangePointsModel(change_point_model=Binseg(), n_bkps=5),
                ),
                "regular_ts",
            ),
            (
                ChangePointsTrendTransform(
                    in_column="target", change_point_model=Binseg(), detrend_model=LinearRegression(), n_bkps=5
                ),
                "regular_ts",
            ),
            (BinsegTrendTransform(in_column="target"), "regular_ts"),
            (LinearTrendTransform(in_column="target"), "regular_ts"),
            (TheilSenTrendTransform(in_column="target"), "regular_ts"),
            (STLTransform(in_column="target", period=7), "regular_ts"),
            (TrendTransform(in_column="target"), "regular_ts"),
            # encoders
            (LabelEncoderTransform(in_column="weekday"), "ts_with_exog"),
            (OneHotEncoderTransform(in_column="weekday"), "ts_with_exog"),
            (MeanSegmentEncoderTransform(), "regular_ts"),
            (SegmentEncoderTransform(), "regular_ts"),
            # feature_selection
            (FilterFeaturesTransform(exclude=["year"]), "ts_with_exog"),
            (GaleShapleyFeatureSelectionTransform(relevance_table=StatisticsRelevanceTable(), top_k=2), "ts_with_exog"),
            (MRMRFeatureSelectionTransform(relevance_table=StatisticsRelevanceTable(), top_k=2), "ts_with_exog"),
            (TreeFeatureSelectionTransform(model=DecisionTreeRegressor(random_state=42), top_k=2), "ts_with_exog"),
            # math
            (AddConstTransform(in_column="target", value=1, inplace=False), "regular_ts"),
            (AddConstTransform(in_column="target", value=1, inplace=True), "regular_ts"),
            (LagTransform(in_column="target", lags=[1, 2, 3]), "regular_ts"),
            (
                LambdaTransform(in_column="target", transform_func=lambda x: x + 1, inplace=False),
                "regular_ts",
            ),
            (
                LambdaTransform(
                    in_column="target",
                    transform_func=lambda x: x + 1,
                    inverse_transform_func=lambda x: x - 1,
                    inplace=True,
                ),
                "regular_ts",
            ),
            (LogTransform(in_column="target", inplace=False), "positive_ts"),
            (LogTransform(in_column="target", inplace=True), "positive_ts"),
            (DifferencingTransform(in_column="target", inplace=False), "regular_ts"),
            (DifferencingTransform(in_column="target", inplace=True), "regular_ts"),
            (MADTransform(in_column="target", window=14), "regular_ts"),
            (MaxTransform(in_column="target", window=14), "regular_ts"),
            (MeanTransform(in_column="target", window=14), "regular_ts"),
            (MedianTransform(in_column="target", window=14), "regular_ts"),
            (MinMaxDifferenceTransform(in_column="target", window=14), "regular_ts"),
            (MinTransform(in_column="target", window=14), "regular_ts"),
            (QuantileTransform(in_column="target", quantile=0.9, window=14), "regular_ts"),
            (StdTransform(in_column="target", window=14), "regular_ts"),
            (SumTransform(in_column="target", window=14), "regular_ts"),
            (BoxCoxTransform(in_column="target", mode="per-segment", inplace=False), "positive_ts"),
            (BoxCoxTransform(in_column="target", mode="per-segment", inplace=True), "positive_ts"),
            (BoxCoxTransform(in_column="target", mode="macro", inplace=False), "positive_ts"),
            (BoxCoxTransform(in_column="target", mode="macro", inplace=True), "positive_ts"),
            (MaxAbsScalerTransform(in_column="target", mode="per-segment", inplace=False), "regular_ts"),
            (MaxAbsScalerTransform(in_column="target", mode="per-segment", inplace=True), "regular_ts"),
            (MaxAbsScalerTransform(in_column="target", mode="macro", inplace=False), "regular_ts"),
            (MaxAbsScalerTransform(in_column="target", mode="macro", inplace=True), "regular_ts"),
            (MinMaxScalerTransform(in_column="target", mode="per-segment", inplace=False), "regular_ts"),
            (MinMaxScalerTransform(in_column="target", mode="per-segment", inplace=True), "regular_ts"),
            (MinMaxScalerTransform(in_column="target", mode="macro", inplace=False), "regular_ts"),
            (MinMaxScalerTransform(in_column="target", mode="macro", inplace=True), "regular_ts"),
            (RobustScalerTransform(in_column="target", mode="per-segment", inplace=False), "regular_ts"),
            (RobustScalerTransform(in_column="target", mode="per-segment", inplace=True), "regular_ts"),
            (RobustScalerTransform(in_column="target", mode="macro", inplace=False), "regular_ts"),
            (RobustScalerTransform(in_column="target", mode="macro", inplace=True), "regular_ts"),
            (StandardScalerTransform(in_column="target", mode="per-segment", inplace=False), "regular_ts"),
            (StandardScalerTransform(in_column="target", mode="per-segment", inplace=True), "regular_ts"),
            (StandardScalerTransform(in_column="target", mode="macro", inplace=False), "regular_ts"),
            (StandardScalerTransform(in_column="target", mode="macro", inplace=True), "regular_ts"),
            (YeoJohnsonTransform(in_column="target", mode="per-segment", inplace=False), "regular_ts"),
            (YeoJohnsonTransform(in_column="target", mode="per-segment", inplace=True), "regular_ts"),
            (YeoJohnsonTransform(in_column="target", mode="macro", inplace=False), "regular_ts"),
            (YeoJohnsonTransform(in_column="target", mode="macro", inplace=True), "regular_ts"),
            # missing_values
            (
                ResampleWithDistributionTransform(
                    in_column="regressor_exog", distribution_column="target", inplace=False
                ),
                "ts_to_resample",
            ),
            (
                ResampleWithDistributionTransform(
                    in_column="regressor_exog", distribution_column="target", inplace=True
                ),
                "ts_to_resample",
            ),
            (TimeSeriesImputerTransform(in_column="target"), "ts_to_fill"),
            # outliers
            (DensityOutliersTransform(in_column="target"), "ts_with_outliers"),
            (MedianOutliersTransform(in_column="target"), "ts_with_outliers"),
            (PredictionIntervalOutliersTransform(in_column="target", model=ProphetModel), "ts_with_outliers"),
            # timestamp
            (DateFlagsTransform(), "regular_ts"),
            (FourierTransform(period=7, order=2), "regular_ts"),
            (HolidayTransform(), "regular_ts"),
            (SpecialDaysTransform(), "regular_ts"),
            (TimeFlagsTransform(), "regular_ts"),
        ],
    )
    def test_transform_future_subset_segments(self, transform, dataset_name, request):
        ts = request.getfixturevalue(dataset_name)
        self._test_transform_future_subset_segments(ts, transform, segments=["segment_2"])


def find_columns_diff(df_before: pd.DataFrame, df_after: pd.DataFrame) -> Tuple[Set[str], Set[str], Set[str]]:
    columns_before_transform = set(df_before.columns)
    columns_after_transform = set(df_after.columns)
    created_columns = columns_after_transform - columns_before_transform
    removed_columns = columns_before_transform - columns_after_transform

    columns_to_check_changes = columns_after_transform.intersection(columns_before_transform)
    changed_columns = set()
    for column in columns_to_check_changes:
        if not df_before[column].equals(df_after[column]):
            changed_columns.add(column)

    return created_columns, removed_columns, changed_columns


class TestTransformTrainNewSegments:
    """Test transform on train part of new segments.

    Expected that transformation creates columns, removes columns and changes values.
    """

    def _test_transform_train_new_segments(self, ts, transform, train_segments, expected_changes):
        # select subset of tsdataset
        train_segments = list(set(train_segments))
        forecast_segments = list(set(ts.segments) - set(train_segments))
        train_ts = select_segments_subset(ts=deepcopy(ts), segments=train_segments)
        test_ts = select_segments_subset(ts=deepcopy(ts), segments=forecast_segments)
        train_df = train_ts.to_pandas()
        test_df = test_ts.to_pandas()

        # fitting
        transform.fit(train_df)

        # transform
        transformed_test_df = transform.transform(test_df.copy())

        # checking
        expected_columns_to_create = expected_changes.get("create", set())
        expected_columns_to_remove = expected_changes.get("remove", set())
        expected_columns_to_change = expected_changes.get("change", set())
        flat_test_df = TSDataset.to_flatten(test_df)
        flat_transformed_test_df = TSDataset.to_flatten(transformed_test_df)
        created_columns, removed_columns, changed_columns = find_columns_diff(flat_test_df, flat_transformed_test_df)

        assert created_columns == expected_columns_to_create
        assert removed_columns == expected_columns_to_remove
        assert changed_columns == expected_columns_to_change

    @pytest.mark.parametrize(
        "transform, dataset_name, expected_changes",
        [
            # encoders
            (LabelEncoderTransform(in_column="weekday", out_column="res"), "ts_with_exog", {"create": {"res"}}),
            (
                OneHotEncoderTransform(in_column="weekday", out_column="res"),
                "ts_with_exog",
                {"create": {"res_0", "res_1", "res_2", "res_3", "res_4", "res_5", "res_6"}},
            ),
            # feature_selection
            (FilterFeaturesTransform(exclude=["year"]), "ts_with_exog", {"remove": {"year"}}),
            # TODO: this should remove only 2 features, wait for fixing [#1097](https://github.com/tinkoff-ai/etna/issues/1097)
            (
                GaleShapleyFeatureSelectionTransform(relevance_table=StatisticsRelevanceTable(), top_k=2),
                "ts_with_exog",
                {"remove": {"weekday", "year", "month", "monthday"}},
            ),
            (
                MRMRFeatureSelectionTransform(relevance_table=StatisticsRelevanceTable(), top_k=2),
                "ts_with_exog",
                {"remove": {"weekday", "monthday"}},
            ),
            (
                TreeFeatureSelectionTransform(model=DecisionTreeRegressor(random_state=42), top_k=2),
                "ts_with_exog",
                {"remove": {"year", "month"}},
            ),
            # math
            (
                AddConstTransform(in_column="target", value=1, inplace=False, out_column="res"),
                "regular_ts",
                {"create": {"res"}},
            ),
            (AddConstTransform(in_column="target", value=1, inplace=True), "regular_ts", {"change": {"target"}}),
            (
                LagTransform(in_column="target", lags=[1, 2, 3], out_column="res"),
                "regular_ts",
                {"create": {"res_1", "res_2", "res_3"}},
            ),
            (
                LambdaTransform(in_column="target", transform_func=lambda x: x + 1, inplace=False, out_column="res"),
                "regular_ts",
                {"create": {"res"}},
            ),
            (
                LambdaTransform(
                    in_column="target",
                    transform_func=lambda x: x + 1,
                    inverse_transform_func=lambda x: x - 1,
                    inplace=True,
                ),
                "regular_ts",
                {"change": {"target"}},
            ),
            (LogTransform(in_column="target", inplace=False, out_column="res"), "positive_ts", {"create": {"res"}}),
            (LogTransform(in_column="target", inplace=True), "positive_ts", {"change": {"target"}}),
            (
                DifferencingTransform(in_column="target", inplace=False, out_column="res"),
                "regular_ts",
                {"create": {"res"}},
            ),
            (DifferencingTransform(in_column="target", inplace=True), "regular_ts", {"change": {"target"}}),
            (MADTransform(in_column="target", window=7, out_column="res"), "regular_ts", {"create": {"res"}}),
            (MaxTransform(in_column="target", window=7, out_column="res"), "regular_ts", {"create": {"res"}}),
            (MeanTransform(in_column="target", window=7, out_column="res"), "regular_ts", {"create": {"res"}}),
            (MedianTransform(in_column="target", window=7, out_column="res"), "regular_ts", {"create": {"res"}}),
            (
                MinMaxDifferenceTransform(in_column="target", window=7, out_column="res"),
                "regular_ts",
                {"create": {"res"}},
            ),
            (MinTransform(in_column="target", window=7, out_column="res"), "regular_ts", {"create": {"res"}}),
            (
                QuantileTransform(in_column="target", quantile=0.9, window=7, out_column="res"),
                "regular_ts",
                {"create": {"res"}},
            ),
            (StdTransform(in_column="target", window=7, out_column="res"), "regular_ts", {"create": {"res"}}),
            (SumTransform(in_column="target", window=7, out_column="res"), "regular_ts", {"create": {"res"}}),
            (
                BoxCoxTransform(in_column="target", mode="macro", inplace=False, out_column="res"),
                "positive_ts",
                {"create": {"res_target"}},
            ),
            (BoxCoxTransform(in_column="target", mode="macro", inplace=True), "positive_ts", {"change": {"target"}}),
            (
                MaxAbsScalerTransform(in_column="target", mode="macro", inplace=False, out_column="res"),
                "regular_ts",
                {"create": {"res_target"}},
            ),
            (
                MaxAbsScalerTransform(in_column="target", mode="macro", inplace=True),
                "regular_ts",
                {"change": {"target"}},
            ),
            (
                MinMaxScalerTransform(in_column="target", mode="macro", inplace=False, out_column="res"),
                "regular_ts",
                {"create": {"res_target"}},
            ),
            (
                MinMaxScalerTransform(in_column="target", mode="macro", inplace=True),
                "regular_ts",
                {"change": {"target"}},
            ),
            (
                RobustScalerTransform(in_column="target", mode="macro", inplace=False, out_column="res"),
                "regular_ts",
                {"create": {"res_target"}},
            ),
            (
                RobustScalerTransform(in_column="target", mode="macro", inplace=True),
                "regular_ts",
                {"change": {"target"}},
            ),
            (
                StandardScalerTransform(in_column="target", mode="macro", inplace=False, out_column="res"),
                "regular_ts",
                {"create": {"res_target"}},
            ),
            (
                StandardScalerTransform(in_column="target", mode="macro", inplace=True),
                "regular_ts",
                {"change": {"target"}},
            ),
            (
                YeoJohnsonTransform(in_column="target", mode="macro", inplace=False, out_column="res"),
                "regular_ts",
                {"create": {"res_target"}},
            ),
            (YeoJohnsonTransform(in_column="target", mode="macro", inplace=True), "regular_ts", {"change": {"target"}}),
            # timestamp
            (
                DateFlagsTransform(out_column="res"),
                "regular_ts",
                {"create": {"res_day_number_in_week", "res_day_number_in_month", "res_is_weekend"}},
            ),
            (
                FourierTransform(period=7, order=2, out_column="res"),
                "regular_ts",
                {"create": {"res_1", "res_2", "res_3", "res_4"}},
            ),
            (HolidayTransform(out_column="res"), "regular_ts", {"create": {"res"}}),
            (
                TimeFlagsTransform(out_column="res"),
                "regular_ts",
                {"create": {"res_minute_in_hour_number", "res_hour_number"}},
            ),
        ],
    )
    def test_transform_train_new_segments(self, transform, dataset_name, expected_changes, request):
        ts = request.getfixturevalue(dataset_name)
        self._test_transform_train_new_segments(
            ts, transform, train_segments=["segment_1", "segment_2"], expected_changes=expected_changes
        )

    @pytest.mark.parametrize(
        "transform, dataset_name",
        [
            # decomposition
            (
                ChangePointsSegmentationTransform(
                    in_column="target",
                    change_point_model=RupturesChangePointsModel(change_point_model=Binseg(), n_bkps=5),
                ),
                "regular_ts",
            ),
            (
                ChangePointsTrendTransform(
                    in_column="target", change_point_model=Binseg(), detrend_model=LinearRegression(), n_bkps=5
                ),
                "regular_ts",
            ),
            (BinsegTrendTransform(in_column="target"), "regular_ts"),
            (LinearTrendTransform(in_column="target"), "regular_ts"),
            (TheilSenTrendTransform(in_column="target"), "regular_ts"),
            (STLTransform(in_column="target", period=7), "regular_ts"),
            (TrendTransform(in_column="target"), "regular_ts"),
            # encoders
            (MeanSegmentEncoderTransform(), "regular_ts"),
            (SegmentEncoderTransform(), "regular_ts"),
            # math
            (BoxCoxTransform(in_column="target", mode="per-segment", inplace=False), "positive_ts"),
            (BoxCoxTransform(in_column="target", mode="per-segment", inplace=True), "positive_ts"),
            (MaxAbsScalerTransform(in_column="target", mode="per-segment", inplace=False), "regular_ts"),
            (MaxAbsScalerTransform(in_column="target", mode="per-segment", inplace=True), "regular_ts"),
            (MinMaxScalerTransform(in_column="target", mode="per-segment", inplace=False), "regular_ts"),
            (MinMaxScalerTransform(in_column="target", mode="per-segment", inplace=True), "regular_ts"),
            (RobustScalerTransform(in_column="target", mode="per-segment", inplace=False), "regular_ts"),
            (RobustScalerTransform(in_column="target", mode="per-segment", inplace=True), "regular_ts"),
            (StandardScalerTransform(in_column="target", mode="per-segment", inplace=False), "regular_ts"),
            (StandardScalerTransform(in_column="target", mode="per-segment", inplace=True), "regular_ts"),
            (YeoJohnsonTransform(in_column="target", mode="per-segment", inplace=False), "regular_ts"),
            (YeoJohnsonTransform(in_column="target", mode="per-segment", inplace=True), "regular_ts"),
            # missing_values
            (
                ResampleWithDistributionTransform(
                    in_column="regressor_exog", distribution_column="target", inplace=False
                ),
                "ts_to_resample",
            ),
            (
                ResampleWithDistributionTransform(
                    in_column="regressor_exog", distribution_column="target", inplace=True
                ),
                "ts_to_resample",
            ),
            (
                TimeSeriesImputerTransform(in_column="target"),
                "ts_to_fill",
            ),
        ],
    )
    def test_transform_train_new_segments_not_implemented(self, transform, dataset_name, request):
        ts = request.getfixturevalue(dataset_name)
        with pytest.raises(NotImplementedError):
            self._test_transform_train_new_segments(
                ts, transform, train_segments=["segment_1", "segment_2"], expected_changes={}
            )

    @to_be_fixed(raises=NotImplementedError, match="Per-segment transforms can't work on new segments")
    @pytest.mark.parametrize(
        "transform, dataset_name",
        [
            # timestamp
            (SpecialDaysTransform(), "regular_ts"),
        ],
    )
    def test_transform_train_new_segments_failed_not_implemented(self, transform, dataset_name, request):
        ts = request.getfixturevalue(dataset_name)
        self._test_transform_train_new_segments(
            ts, transform, train_segments=["segment_1", "segment_2"], expected_changes={}
        )

    @to_be_fixed(raises=Exception)
    @pytest.mark.parametrize(
        "transform, dataset_name, expected_changes",
        [
            # outliers
            # TODO: error should be understandable, not like now
            (DensityOutliersTransform(in_column="target"), "ts_with_outliers", {}),
            (MedianOutliersTransform(in_column="target"), "ts_with_outliers", {}),
            (PredictionIntervalOutliersTransform(in_column="target", model=ProphetModel), "ts_with_outliers", {}),
        ],
    )
    def test_transform_train_new_segments_failed_error(self, transform, dataset_name, expected_changes, request):
        ts = request.getfixturevalue(dataset_name)
        self._test_transform_train_new_segments(
            ts, transform, train_segments=["segment_1", "segment_2"], expected_changes=expected_changes
        )


class TestTransformFutureNewSegments:
    """Test transform on future part of new segments.

    Expected that transformation creates columns, removes columns and changes values.
    """

    def _test_transform_future_new_segments(self, ts, transform, train_segments, expected_changes, horizon=7):
        # select subset of tsdataset
        train_segments = list(set(train_segments))
        forecast_segments = list(set(ts.segments) - set(train_segments))
        train_ts = select_segments_subset(ts=deepcopy(ts), segments=train_segments)
        test_ts_without_transform = select_segments_subset(ts=deepcopy(ts), segments=forecast_segments)
        test_ts_with_transform = select_segments_subset(ts=deepcopy(ts), segments=forecast_segments)
        test_ts_without_transform.transforms = []
        test_ts_with_transform.transforms = [transform]
        train_df = train_ts.to_pandas()

        # fitting
        transform.fit(train_df)

        # prepare df without transform
        non_transformed_test_ts = test_ts_without_transform.make_future(future_steps=horizon)
        non_transformed_test_df = non_transformed_test_ts.to_pandas()

        # transform
        transformed_test_ts = test_ts_with_transform.make_future(future_steps=horizon)
        transformed_test_df = transformed_test_ts.to_pandas()

        # checking
        expected_columns_to_create = expected_changes.get("create", set())
        expected_columns_to_remove = expected_changes.get("remove", set())
        expected_columns_to_change = expected_changes.get("change", set())
        flat_non_transformed_test_df = TSDataset.to_flatten(non_transformed_test_df)
        flat_transformed_test_df = TSDataset.to_flatten(transformed_test_df)
        created_columns, removed_columns, changed_columns = find_columns_diff(
            flat_non_transformed_test_df, flat_transformed_test_df
        )

        assert created_columns == expected_columns_to_create
        assert removed_columns == expected_columns_to_remove
        assert changed_columns == expected_columns_to_change

    @pytest.mark.parametrize(
        "transform, dataset_name, expected_changes",
        [
            # encoders
            (LabelEncoderTransform(in_column="weekday", out_column="res"), "ts_with_exog", {"create": {"res"}}),
            (
                OneHotEncoderTransform(in_column="weekday", out_column="res"),
                "ts_with_exog",
                {"create": {"res_0", "res_1", "res_2", "res_3", "res_4", "res_5", "res_6"}},
            ),
            # feature_selection
            (FilterFeaturesTransform(exclude=["year"]), "ts_with_exog", {"remove": {"year"}}),
            # TODO: this should remove only 2 features
            (
                GaleShapleyFeatureSelectionTransform(relevance_table=StatisticsRelevanceTable(), top_k=2),
                "ts_with_exog",
                {"remove": {"weekday", "year", "month", "monthday"}},
            ),
            (
                MRMRFeatureSelectionTransform(relevance_table=StatisticsRelevanceTable(), top_k=2),
                "ts_with_exog",
                {"remove": {"weekday", "monthday"}},
            ),
            (
                TreeFeatureSelectionTransform(model=DecisionTreeRegressor(random_state=42), top_k=2),
                "ts_with_exog",
                {"remove": {"year", "month"}},
            ),
            # math
            (
                AddConstTransform(in_column="target", value=1, inplace=False, out_column="res"),
                "regular_ts",
                {"create": {"res"}},
            ),
            (AddConstTransform(in_column="target", value=1, inplace=True), "regular_ts", {}),
            (
                LagTransform(in_column="target", lags=[1, 2, 3], out_column="res"),
                "regular_ts",
                {"create": {"res_1", "res_2", "res_3"}},
            ),
            (
                LambdaTransform(in_column="target", transform_func=lambda x: x + 1, inplace=False, out_column="res"),
                "regular_ts",
                {"create": {"res"}},
            ),
            (
                LambdaTransform(
                    in_column="target",
                    transform_func=lambda x: x + 1,
                    inverse_transform_func=lambda x: x - 1,
                    inplace=True,
                ),
                "regular_ts",
                {},
            ),
            (LogTransform(in_column="target", inplace=False, out_column="res"), "positive_ts", {"create": {"res"}}),
            (LogTransform(in_column="target", inplace=True), "positive_ts", {}),
            (
                DifferencingTransform(in_column="target", inplace=False, out_column="res"),
                "regular_ts",
                {"create": {"res"}},
            ),
            (DifferencingTransform(in_column="target", inplace=True), "regular_ts", {}),
            (MADTransform(in_column="target", window=14, out_column="res"), "regular_ts", {"create": {"res"}}),
            (MaxTransform(in_column="target", window=14, out_column="res"), "regular_ts", {"create": {"res"}}),
            (MeanTransform(in_column="target", window=14, out_column="res"), "regular_ts", {"create": {"res"}}),
            (MedianTransform(in_column="target", window=14, out_column="res"), "regular_ts", {"create": {"res"}}),
            (
                MinMaxDifferenceTransform(in_column="target", window=14, out_column="res"),
                "regular_ts",
                {"create": {"res"}},
            ),
            (MinTransform(in_column="target", window=14, out_column="res"), "regular_ts", {"create": {"res"}}),
            (
                QuantileTransform(in_column="target", quantile=0.9, window=14, out_column="res"),
                "regular_ts",
                {"create": {"res"}},
            ),
            (StdTransform(in_column="target", window=14, out_column="res"), "regular_ts", {"create": {"res"}}),
            (SumTransform(in_column="target", window=14, out_column="res"), "regular_ts", {"create": {"res"}}),
            (
                BoxCoxTransform(in_column="target", mode="macro", inplace=False, out_column="res"),
                "positive_ts",
                {"create": {"res_target"}},
            ),
            (BoxCoxTransform(in_column="target", mode="macro", inplace=True), "positive_ts", {}),
            (
                MaxAbsScalerTransform(in_column="target", mode="macro", inplace=False, out_column="res"),
                "regular_ts",
                {"create": {"res_target"}},
            ),
            (
                MaxAbsScalerTransform(in_column="target", mode="macro", inplace=True),
                "regular_ts",
                {},
            ),
            (
                MinMaxScalerTransform(in_column="target", mode="macro", inplace=False, out_column="res"),
                "regular_ts",
                {"create": {"res_target"}},
            ),
            (
                MinMaxScalerTransform(in_column="target", mode="macro", inplace=True),
                "regular_ts",
                {},
            ),
            (
                RobustScalerTransform(in_column="target", mode="macro", inplace=False, out_column="res"),
                "regular_ts",
                {"create": {"res_target"}},
            ),
            (
                RobustScalerTransform(in_column="target", mode="macro", inplace=True),
                "regular_ts",
                {},
            ),
            (
                StandardScalerTransform(in_column="target", mode="macro", inplace=False, out_column="res"),
                "regular_ts",
                {"create": {"res_target"}},
            ),
            (
                StandardScalerTransform(in_column="target", mode="macro", inplace=True),
                "regular_ts",
                {},
            ),
            (
                YeoJohnsonTransform(in_column="target", mode="macro", inplace=False, out_column="res"),
                "regular_ts",
                {"create": {"res_target"}},
            ),
            (YeoJohnsonTransform(in_column="target", mode="macro", inplace=True), "regular_ts", {}),
            # timestamp
            (
                DateFlagsTransform(out_column="res"),
                "regular_ts",
                {"create": {"res_day_number_in_week", "res_day_number_in_month", "res_is_weekend"}},
            ),
            (
                FourierTransform(period=7, order=2, out_column="res"),
                "regular_ts",
                {"create": {"res_1", "res_2", "res_3", "res_4"}},
            ),
            (HolidayTransform(out_column="res"), "regular_ts", {"create": {"res"}}),
            (
                TimeFlagsTransform(out_column="res"),
                "regular_ts",
                {"create": {"res_minute_in_hour_number", "res_hour_number"}},
            ),
        ],
    )
    def test_transform_future_new_segments(self, transform, dataset_name, expected_changes, request):
        ts = request.getfixturevalue(dataset_name)
        self._test_transform_future_new_segments(
            ts, transform, train_segments=["segment_1", "segment_2"], expected_changes=expected_changes
        )

    @pytest.mark.parametrize(
        "transform, dataset_name",
        [
            # decomposition
            (
                ChangePointsSegmentationTransform(
                    in_column="target",
                    change_point_model=RupturesChangePointsModel(change_point_model=Binseg(), n_bkps=5),
                ),
                "regular_ts",
            ),
            (
                ChangePointsTrendTransform(
                    in_column="target", change_point_model=Binseg(), detrend_model=LinearRegression(), n_bkps=5
                ),
                "regular_ts",
            ),
            (BinsegTrendTransform(in_column="target"), "regular_ts"),
            (LinearTrendTransform(in_column="target"), "regular_ts"),
            (TheilSenTrendTransform(in_column="target"), "regular_ts"),
            (STLTransform(in_column="target", period=7), "regular_ts"),
            (TrendTransform(in_column="target"), "regular_ts"),
            # encoders
            (MeanSegmentEncoderTransform(), "regular_ts"),
            (SegmentEncoderTransform(), "regular_ts"),
            # math
            (BoxCoxTransform(in_column="target", mode="per-segment", inplace=False), "positive_ts"),
            (BoxCoxTransform(in_column="target", mode="per-segment", inplace=True), "positive_ts"),
            (MaxAbsScalerTransform(in_column="target", mode="per-segment", inplace=False), "regular_ts"),
            (MaxAbsScalerTransform(in_column="target", mode="per-segment", inplace=True), "regular_ts"),
            (MinMaxScalerTransform(in_column="target", mode="per-segment", inplace=False), "regular_ts"),
            (MinMaxScalerTransform(in_column="target", mode="per-segment", inplace=True), "regular_ts"),
            (RobustScalerTransform(in_column="target", mode="per-segment", inplace=False), "regular_ts"),
            (RobustScalerTransform(in_column="target", mode="per-segment", inplace=True), "regular_ts"),
            (StandardScalerTransform(in_column="target", mode="per-segment", inplace=False), "regular_ts"),
            (StandardScalerTransform(in_column="target", mode="per-segment", inplace=True), "regular_ts"),
            (YeoJohnsonTransform(in_column="target", mode="per-segment", inplace=False), "regular_ts"),
            (YeoJohnsonTransform(in_column="target", mode="per-segment", inplace=True), "regular_ts"),
            # missing_values
            (
                ResampleWithDistributionTransform(
                    in_column="regressor_exog", distribution_column="target", inplace=False
                ),
                "ts_to_resample",
            ),
            (
                ResampleWithDistributionTransform(
                    in_column="regressor_exog", distribution_column="target", inplace=True
                ),
                "ts_to_resample",
            ),
            (
                TimeSeriesImputerTransform(in_column="target"),
                "ts_to_fill",
            ),
        ],
    )
    def test_transform_future_new_segments_not_implemented(self, transform, dataset_name, request):
        ts = request.getfixturevalue(dataset_name)
        with pytest.raises(NotImplementedError):
            self._test_transform_future_new_segments(
                ts, transform, train_segments=["segment_1", "segment_2"], expected_changes={}
            )

    @to_be_fixed(raises=NotImplementedError, match="Per-segment transforms can't work on new segments")
    @pytest.mark.parametrize(
        "transform, dataset_name",
        [
            # timestamp
            (SpecialDaysTransform(), "regular_ts"),
        ],
    )
    def test_transform_future_new_segments_failed_not_implemented(self, transform, dataset_name, request):
        ts = request.getfixturevalue(dataset_name)
        self._test_transform_future_new_segments(
            ts, transform, train_segments=["segment_1", "segment_2"], expected_changes={}
        )

    @to_be_fixed(raises=Exception)
    @pytest.mark.parametrize(
        "transform, dataset_name, expected_changes",
        [
            # outliers
            # TODO: error should be understandable, not like now
            (DensityOutliersTransform(in_column="target"), "ts_with_outliers", {}),
            (MedianOutliersTransform(in_column="target"), "ts_with_outliers", {}),
            (PredictionIntervalOutliersTransform(in_column="target", model=ProphetModel), "ts_with_outliers", {}),
        ],
    )
    def test_transform_future_new_segments_failed_error(self, transform, dataset_name, expected_changes, request):
        ts = request.getfixturevalue(dataset_name)
        self._test_transform_future_new_segments(
            ts, transform, train_segments=["segment_1", "segment_2"], expected_changes=expected_changes
        )


class TestTransformFutureWithTarget:
    """Test transform on future dataset with known target.

    Expected that transformation creates columns, removes columns and changes values.
    """

    def _test_transform_future_with_target(self, ts, transform, expected_changes, gap_size=7, transform_size=50):
        # select subset of tsdataset
        history_ts, future_full_ts = ts.train_test_split(test_size=gap_size + transform_size)
        _, future_suffix_ts = future_full_ts.train_test_split(test_size=transform_size)
        train_df = history_ts.to_pandas()
        future_suffix_df = future_suffix_ts.to_pandas()

        # fitting
        transform.fit(train_df)

        # transform
        transformed_future_suffix_df = transform.transform(future_suffix_df.copy())

        # checking
        expected_columns_to_create = expected_changes.get("create", set())
        expected_columns_to_remove = expected_changes.get("remove", set())
        expected_columns_to_change = expected_changes.get("change", set())
        flat_future_suffix_df = TSDataset.to_flatten(future_suffix_df)
        flat_transformed_future_suffix_df = TSDataset.to_flatten(transformed_future_suffix_df)
        created_columns, removed_columns, changed_columns = find_columns_diff(
            flat_future_suffix_df, flat_transformed_future_suffix_df
        )

        assert created_columns == expected_columns_to_create
        assert removed_columns == expected_columns_to_remove
        assert changed_columns == expected_columns_to_change

    @pytest.mark.parametrize(
        "transform, dataset_name, expected_changes",
        [
            # decomposition
            (
                ChangePointsSegmentationTransform(
                    in_column="target",
                    change_point_model=RupturesChangePointsModel(change_point_model=Binseg(), n_bkps=5),
                    out_column="res",
                ),
                "regular_ts",
                {"create": {"res"}},
            ),
            (
                ChangePointsTrendTransform(
                    in_column="target", change_point_model=Binseg(), detrend_model=LinearRegression(), n_bkps=5
                ),
                "regular_ts",
                {"change": {"target"}},
            ),
            (BinsegTrendTransform(in_column="target"), "regular_ts", {"change": {"target"}}),
            (LinearTrendTransform(in_column="target"), "regular_ts", {"change": {"target"}}),
            (TheilSenTrendTransform(in_column="target"), "regular_ts", {"change": {"target"}}),
            (STLTransform(in_column="target", period=7), "regular_ts", {"change": {"target"}}),
            (TrendTransform(in_column="target", out_column="res"), "regular_ts", {"create": {"res"}}),
            # encoders
            (LabelEncoderTransform(in_column="weekday", out_column="res"), "ts_with_exog", {"create": {"res"}}),
            (
                OneHotEncoderTransform(in_column="weekday", out_column="res"),
                "ts_with_exog",
                {"create": {"res_0", "res_1", "res_2", "res_3", "res_4", "res_5", "res_6"}},
            ),
            (MeanSegmentEncoderTransform(), "regular_ts", {"create": {"segment_mean"}}),
            (SegmentEncoderTransform(), "regular_ts", {"create": {"segment_code"}}),
            # feature_selection
            (FilterFeaturesTransform(exclude=["year"]), "ts_with_exog", {"remove": {"year"}}),
            # TODO: this should remove only 2 features
            (
                GaleShapleyFeatureSelectionTransform(relevance_table=StatisticsRelevanceTable(), top_k=2),
                "ts_with_exog",
                {"remove": {"month", "year"}},
            ),
            (
                MRMRFeatureSelectionTransform(relevance_table=StatisticsRelevanceTable(), top_k=2),
                "ts_with_exog",
                {"remove": {"weekday", "monthday"}},
            ),
            (
                TreeFeatureSelectionTransform(model=DecisionTreeRegressor(random_state=42), top_k=2),
                "ts_with_exog",
                {"remove": {"year", "month"}},
            ),
            # math
            (
                AddConstTransform(in_column="target", value=1, inplace=False, out_column="res"),
                "regular_ts",
                {"create": {"res"}},
            ),
            (AddConstTransform(in_column="target", value=1, inplace=True), "regular_ts", {"change": {"target"}}),
            (
                LagTransform(in_column="target", lags=[1, 2, 3], out_column="res"),
                "regular_ts",
                {"create": {"res_1", "res_2", "res_3"}},
            ),
            (
                LambdaTransform(in_column="target", transform_func=lambda x: x + 1, inplace=False, out_column="res"),
                "regular_ts",
                {"create": {"res"}},
            ),
            (
                LambdaTransform(
                    in_column="target",
                    transform_func=lambda x: x + 1,
                    inverse_transform_func=lambda x: x - 1,
                    inplace=True,
                ),
                "regular_ts",
                {"change": {"target"}},
            ),
            (LogTransform(in_column="target", inplace=False, out_column="res"), "positive_ts", {"create": {"res"}}),
            (LogTransform(in_column="target", inplace=True), "positive_ts", {"change": {"target"}}),
            (
                DifferencingTransform(in_column="target", inplace=False, out_column="res"),
                "regular_ts",
                {"create": {"res"}},
            ),
            (DifferencingTransform(in_column="target", inplace=True), "regular_ts", {"change": {"target"}}),
            (MADTransform(in_column="target", window=7, out_column="res"), "regular_ts", {"create": {"res"}}),
            (MaxTransform(in_column="target", window=7, out_column="res"), "regular_ts", {"create": {"res"}}),
            (MeanTransform(in_column="target", window=7, out_column="res"), "regular_ts", {"create": {"res"}}),
            (MedianTransform(in_column="target", window=7, out_column="res"), "regular_ts", {"create": {"res"}}),
            (
                MinMaxDifferenceTransform(in_column="target", window=7, out_column="res"),
                "regular_ts",
                {"create": {"res"}},
            ),
            (MinTransform(in_column="target", window=7, out_column="res"), "regular_ts", {"create": {"res"}}),
            (
                QuantileTransform(in_column="target", quantile=0.9, window=7, out_column="res"),
                "regular_ts",
                {"create": {"res"}},
            ),
            (StdTransform(in_column="target", window=7, out_column="res"), "regular_ts", {"create": {"res"}}),
            (SumTransform(in_column="target", window=7, out_column="res"), "regular_ts", {"create": {"res"}}),
            (
                BoxCoxTransform(in_column="target", mode="per-segment", inplace=False, out_column="res"),
                "positive_ts",
                {"create": {"res_target"}},
            ),
            (
                BoxCoxTransform(in_column="target", mode="per-segment", inplace=True),
                "positive_ts",
                {"change": {"target"}},
            ),
            (
                BoxCoxTransform(in_column="target", mode="macro", inplace=False, out_column="res"),
                "positive_ts",
                {"create": {"res_target"}},
            ),
            (BoxCoxTransform(in_column="target", mode="macro", inplace=True), "positive_ts", {"change": {"target"}}),
            (
                MaxAbsScalerTransform(in_column="target", mode="per-segment", inplace=False, out_column="res"),
                "regular_ts",
                {"create": {"res_target"}},
            ),
            (
                MaxAbsScalerTransform(in_column="target", mode="per-segment", inplace=True),
                "regular_ts",
                {"change": {"target"}},
            ),
            (
                MaxAbsScalerTransform(in_column="target", mode="macro", inplace=False, out_column="res"),
                "regular_ts",
                {"create": {"res_target"}},
            ),
            (
                MaxAbsScalerTransform(in_column="target", mode="macro", inplace=True),
                "regular_ts",
                {"change": {"target"}},
            ),
            (
                MinMaxScalerTransform(in_column="target", mode="per-segment", inplace=False, out_column="res"),
                "regular_ts",
                {"create": {"res_target"}},
            ),
            (
                MinMaxScalerTransform(in_column="target", mode="per-segment", inplace=True),
                "regular_ts",
                {"change": {"target"}},
            ),
            (
                MinMaxScalerTransform(in_column="target", mode="macro", inplace=False, out_column="res"),
                "regular_ts",
                {"create": {"res_target"}},
            ),
            (
                MinMaxScalerTransform(in_column="target", mode="macro", inplace=True),
                "regular_ts",
                {"change": {"target"}},
            ),
            (
                RobustScalerTransform(in_column="target", mode="per-segment", inplace=False, out_column="res"),
                "regular_ts",
                {"create": {"res_target"}},
            ),
            (
                RobustScalerTransform(in_column="target", mode="per-segment", inplace=True),
                "regular_ts",
                {"change": {"target"}},
            ),
            (
                RobustScalerTransform(in_column="target", mode="macro", inplace=False, out_column="res"),
                "regular_ts",
                {"create": {"res_target"}},
            ),
            (
                RobustScalerTransform(in_column="target", mode="macro", inplace=True),
                "regular_ts",
                {"change": {"target"}},
            ),
            (
                StandardScalerTransform(in_column="target", mode="per-segment", inplace=False, out_column="res"),
                "regular_ts",
                {"create": {"res_target"}},
            ),
            (
                StandardScalerTransform(in_column="target", mode="per-segment", inplace=True),
                "regular_ts",
                {"change": {"target"}},
            ),
            (
                StandardScalerTransform(in_column="target", mode="macro", inplace=False, out_column="res"),
                "regular_ts",
                {"create": {"res_target"}},
            ),
            (
                StandardScalerTransform(in_column="target", mode="macro", inplace=True),
                "regular_ts",
                {"change": {"target"}},
            ),
            (
                YeoJohnsonTransform(in_column="target", mode="per-segment", inplace=False, out_column="res"),
                "regular_ts",
                {"create": {"res_target"}},
            ),
            (
                YeoJohnsonTransform(in_column="target", mode="per-segment", inplace=True),
                "regular_ts",
                {"change": {"target"}},
            ),
            (
                YeoJohnsonTransform(in_column="target", mode="macro", inplace=False, out_column="res"),
                "regular_ts",
                {"create": {"res_target"}},
            ),
            (YeoJohnsonTransform(in_column="target", mode="macro", inplace=True), "regular_ts", {"change": {"target"}}),
            # missing_values
            (
                ResampleWithDistributionTransform(
                    in_column="regressor_exog", distribution_column="target", inplace=False, out_column="res"
                ),
                "ts_to_resample",
                {"create": {"res"}},
            ),
            (
                ResampleWithDistributionTransform(
                    in_column="regressor_exog", distribution_column="target", inplace=True
                ),
                "ts_to_resample",
                {"change": {"regressor_exog"}},
            ),
            (
                # this behaviour can be unexpected for someone
                TimeSeriesImputerTransform(in_column="target"),
                "ts_to_fill",
                {},
            ),
            # timestamp
            (
                DateFlagsTransform(out_column="res"),
                "regular_ts",
                {"create": {"res_day_number_in_week", "res_day_number_in_month", "res_is_weekend"}},
            ),
            (
                FourierTransform(period=7, order=2, out_column="res"),
                "regular_ts",
                {"create": {"res_1", "res_2", "res_3", "res_4"}},
            ),
            (HolidayTransform(out_column="res"), "regular_ts", {"create": {"res"}}),
            (
                TimeFlagsTransform(out_column="res"),
                "regular_ts",
                {"create": {"res_minute_in_hour_number", "res_hour_number"}},
            ),
            (SpecialDaysTransform(), "regular_ts", {"create": {"anomaly_weekdays", "anomaly_monthdays"}}),
        ],
    )
    def test_transform_future_with_target(self, transform, dataset_name, expected_changes, request):
        ts = request.getfixturevalue(dataset_name)
        self._test_transform_future_with_target(ts, transform, expected_changes=expected_changes)

    @to_be_fixed(raises=Exception)
    @pytest.mark.parametrize(
        "transform, dataset_name, expected_changes",
        [
            # outliers
            (DensityOutliersTransform(in_column="target"), "ts_with_outliers", {}),
            (MedianOutliersTransform(in_column="target"), "ts_with_outliers", {}),
            (PredictionIntervalOutliersTransform(in_column="target", model=ProphetModel), "ts_with_outliers", {}),
        ],
    )
    def test_transform_future_with_target_failed_error(self, transform, dataset_name, expected_changes, request):
        ts = request.getfixturevalue(dataset_name)
        self._test_transform_future_with_target(ts, transform, expected_changes=expected_changes)


class TestTransformFutureWithoutTarget:
    """Test transform on future dataset with unknown target.

    Expected that transformation creates columns, removes columns and changes values.
    """

    def _test_transform_future_without_target(self, ts, transform, expected_changes, gap_size=28, transform_size=7):
        # select subset of tsdataset
        history_ts, test_ts = ts.train_test_split(test_size=gap_size)
        test_ts_without_transform = test_ts
        test_ts_with_transform = deepcopy(test_ts)
        test_ts_without_transform.transforms = []
        test_ts_with_transform.transforms = [transform]
        train_df = history_ts.to_pandas()

        # fitting
        transform.fit(train_df)

        # prepare df without transform
        non_transformed_test_ts = test_ts_without_transform.make_future(future_steps=transform_size)
        non_transformed_test_df = non_transformed_test_ts.to_pandas()

        # transform
        transformed_test_ts = test_ts_with_transform.make_future(future_steps=transform_size)
        transformed_test_df = transformed_test_ts.to_pandas()

        # checking
        expected_columns_to_create = expected_changes.get("create", set())
        expected_columns_to_remove = expected_changes.get("remove", set())
        expected_columns_to_change = expected_changes.get("change", set())
        flat_non_transformed_test_df = TSDataset.to_flatten(non_transformed_test_df)
        flat_transformed_test_df = TSDataset.to_flatten(transformed_test_df)
        created_columns, removed_columns, changed_columns = find_columns_diff(
            flat_non_transformed_test_df, flat_transformed_test_df
        )

        assert created_columns == expected_columns_to_create
        assert removed_columns == expected_columns_to_remove
        assert changed_columns == expected_columns_to_change

    @pytest.mark.parametrize(
        "transform, dataset_name, expected_changes",
        [
            # decomposition
            (
                ChangePointsSegmentationTransform(
                    in_column="target",
                    change_point_model=RupturesChangePointsModel(change_point_model=Binseg(), n_bkps=5),
                    out_column="res",
                ),
                "regular_ts",
                {"create": {"res"}},
            ),
            (
                ChangePointsTrendTransform(
                    in_column="target", change_point_model=Binseg(), detrend_model=LinearRegression(), n_bkps=5
                ),
                "regular_ts",
                {},
            ),
            (BinsegTrendTransform(in_column="target"), "regular_ts", {}),
            (LinearTrendTransform(in_column="target"), "regular_ts", {}),
            (TheilSenTrendTransform(in_column="target"), "regular_ts", {}),
            (STLTransform(in_column="target", period=7), "regular_ts", {}),
            (TrendTransform(in_column="target", out_column="res"), "regular_ts", {"create": {"res"}}),
            # encoders
            (LabelEncoderTransform(in_column="weekday", out_column="res"), "ts_with_exog", {"create": {"res"}}),
            (
                OneHotEncoderTransform(in_column="weekday", out_column="res"),
                "ts_with_exog",
                {"create": {"res_0", "res_1", "res_2", "res_3", "res_4", "res_5", "res_6"}},
            ),
            (MeanSegmentEncoderTransform(), "regular_ts", {"create": {"segment_mean"}}),
            (SegmentEncoderTransform(), "regular_ts", {"create": {"segment_code"}}),
            # feature_selection
            (FilterFeaturesTransform(exclude=["year"]), "ts_with_exog", {"remove": {"year"}}),
            (
                GaleShapleyFeatureSelectionTransform(relevance_table=StatisticsRelevanceTable(), top_k=2),
                "ts_with_exog",
                {"remove": {"month", "year"}},
            ),
            (
                MRMRFeatureSelectionTransform(relevance_table=StatisticsRelevanceTable(), top_k=2),
                "ts_with_exog",
                {"remove": {"weekday", "monthday"}},
            ),
            (
                TreeFeatureSelectionTransform(model=DecisionTreeRegressor(random_state=42), top_k=2),
                "ts_with_exog",
                {"remove": {"year", "month"}},
            ),
            # math
            (
                AddConstTransform(in_column="target", value=1, inplace=False, out_column="res"),
                "regular_ts",
                {"create": {"res"}},
            ),
            (AddConstTransform(in_column="target", value=1, inplace=True), "regular_ts", {}),
            (
                LagTransform(in_column="target", lags=[1, 2, 3], out_column="res"),
                "regular_ts",
                {"create": {"res_1", "res_2", "res_3"}},
            ),
            (
                LambdaTransform(in_column="target", transform_func=lambda x: x + 1, inplace=False, out_column="res"),
                "regular_ts",
                {"create": {"res"}},
            ),
            (
                LambdaTransform(
                    in_column="target",
                    transform_func=lambda x: x + 1,
                    inverse_transform_func=lambda x: x - 1,
                    inplace=True,
                ),
                "regular_ts",
                {},
            ),
            (LogTransform(in_column="target", inplace=False, out_column="res"), "positive_ts", {"create": {"res"}}),
            (LogTransform(in_column="target", inplace=True), "positive_ts", {}),
            (
                DifferencingTransform(in_column="target", inplace=False, out_column="res"),
                "regular_ts",
                {"create": {"res"}},
            ),
            (DifferencingTransform(in_column="target", inplace=True), "regular_ts", {}),
            (MADTransform(in_column="target", window=14, out_column="res"), "regular_ts", {"create": {"res"}}),
            (MaxTransform(in_column="target", window=14, out_column="res"), "regular_ts", {"create": {"res"}}),
            (MeanTransform(in_column="target", window=14, out_column="res"), "regular_ts", {"create": {"res"}}),
            (MedianTransform(in_column="target", window=14, out_column="res"), "regular_ts", {"create": {"res"}}),
            (
                MinMaxDifferenceTransform(in_column="target", window=14, out_column="res"),
                "regular_ts",
                {"create": {"res"}},
            ),
            (MinTransform(in_column="target", window=14, out_column="res"), "regular_ts", {"create": {"res"}}),
            (
                QuantileTransform(in_column="target", quantile=0.9, window=14, out_column="res"),
                "regular_ts",
                {"create": {"res"}},
            ),
            (StdTransform(in_column="target", window=14, out_column="res"), "regular_ts", {"create": {"res"}}),
            (SumTransform(in_column="target", window=14, out_column="res"), "regular_ts", {"create": {"res"}}),
            (
                BoxCoxTransform(in_column="target", mode="macro", inplace=False, out_column="res"),
                "positive_ts",
                {"create": {"res_target"}},
            ),
            (
                BoxCoxTransform(in_column="target", mode="per-segment", inplace=False, out_column="res"),
                "positive_ts",
                {"create": {"res_target"}},
            ),
            (BoxCoxTransform(in_column="target", mode="per-segment", inplace=True), "positive_ts", {}),
            (BoxCoxTransform(in_column="target", mode="macro", inplace=True), "positive_ts", {}),
            (
                MaxAbsScalerTransform(in_column="target", mode="per-segment", inplace=False, out_column="res"),
                "regular_ts",
                {"create": {"res_target"}},
            ),
            (MaxAbsScalerTransform(in_column="target", mode="per-segment", inplace=True), "regular_ts", {}),
            (
                MaxAbsScalerTransform(in_column="target", mode="macro", inplace=False, out_column="res"),
                "regular_ts",
                {"create": {"res_target"}},
            ),
            (
                MaxAbsScalerTransform(in_column="target", mode="macro", inplace=True),
                "regular_ts",
                {},
            ),
            (
                MinMaxScalerTransform(in_column="target", mode="per-segment", inplace=False, out_column="res"),
                "regular_ts",
                {"create": {"res_target"}},
            ),
            (MinMaxScalerTransform(in_column="target", mode="per-segment", inplace=True), "regular_ts", {}),
            (
                MinMaxScalerTransform(in_column="target", mode="macro", inplace=False, out_column="res"),
                "regular_ts",
                {"create": {"res_target"}},
            ),
            (
                MinMaxScalerTransform(in_column="target", mode="macro", inplace=True),
                "regular_ts",
                {},
            ),
            (
                RobustScalerTransform(in_column="target", mode="per-segment", inplace=False, out_column="res"),
                "regular_ts",
                {"create": {"res_target"}},
            ),
            (RobustScalerTransform(in_column="target", mode="per-segment", inplace=True), "regular_ts", {}),
            (
                RobustScalerTransform(in_column="target", mode="macro", inplace=False, out_column="res"),
                "regular_ts",
                {"create": {"res_target"}},
            ),
            (
                RobustScalerTransform(in_column="target", mode="macro", inplace=True),
                "regular_ts",
                {},
            ),
            (
                StandardScalerTransform(in_column="target", mode="per-segment", inplace=False, out_column="res"),
                "regular_ts",
                {"create": {"res_target"}},
            ),
            (StandardScalerTransform(in_column="target", mode="per-segment", inplace=True), "regular_ts", {}),
            (
                StandardScalerTransform(in_column="target", mode="macro", inplace=False, out_column="res"),
                "regular_ts",
                {"create": {"res_target"}},
            ),
            (
                StandardScalerTransform(in_column="target", mode="macro", inplace=True),
                "regular_ts",
                {},
            ),
            (
                YeoJohnsonTransform(in_column="target", mode="per-segment", inplace=False, out_column="res"),
                "regular_ts",
                {"create": {"res_target"}},
            ),
            (YeoJohnsonTransform(in_column="target", mode="per-segment", inplace=True), "regular_ts", {}),
            (
                YeoJohnsonTransform(in_column="target", mode="macro", inplace=False, out_column="res"),
                "regular_ts",
                {"create": {"res_target"}},
            ),
            (YeoJohnsonTransform(in_column="target", mode="macro", inplace=True), "regular_ts", {}),
            # missing_values
            (
                ResampleWithDistributionTransform(
                    in_column="regressor_exog", distribution_column="target", inplace=False, out_column="res"
                ),
                "ts_to_resample",
                {"create": {"res"}},
            ),
            (
                ResampleWithDistributionTransform(
                    in_column="regressor_exog", distribution_column="target", inplace=True
                ),
                "ts_to_resample",
                {"change": {"regressor_exog"}},
            ),
            (
                # this behaviour can be unexpected for someone
                TimeSeriesImputerTransform(in_column="target"),
                "ts_to_fill",
                {},
            ),
            # outliers
            (DensityOutliersTransform(in_column="target"), "ts_with_outliers", {}),
            (MedianOutliersTransform(in_column="target"), "ts_with_outliers", {}),
            (PredictionIntervalOutliersTransform(in_column="target", model=ProphetModel), "ts_with_outliers", {}),
            # timestamp
            (
                DateFlagsTransform(out_column="res"),
                "regular_ts",
                {"create": {"res_day_number_in_week", "res_day_number_in_month", "res_is_weekend"}},
            ),
            (
                FourierTransform(period=7, order=2, out_column="res"),
                "regular_ts",
                {"create": {"res_1", "res_2", "res_3", "res_4"}},
            ),
            (HolidayTransform(out_column="res"), "regular_ts", {"create": {"res"}}),
            (
                TimeFlagsTransform(out_column="res"),
                "regular_ts",
                {"create": {"res_minute_in_hour_number", "res_hour_number"}},
            ),
            (SpecialDaysTransform(), "regular_ts", {"create": {"anomaly_weekdays", "anomaly_monthdays"}}),
        ],
    )
    def test_transform_future_without_target(self, transform, dataset_name, expected_changes, request):
        ts = request.getfixturevalue(dataset_name)
        self._test_transform_future_without_target(ts, transform, expected_changes=expected_changes)
