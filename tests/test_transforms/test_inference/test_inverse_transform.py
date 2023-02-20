from copy import deepcopy
from typing import Set
from typing import Tuple

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




class TestInverseTransformTrainSubsetSegments:
    """Test inverse transform on train part of subset of segments.

    Expected that inverse transformation on subset of segments match subset of inverse transformation on full dataset.
    """

    def _test_inverse_transform_train_subset_segments(self, ts, transform, segments):
        # select subset of tsdataset
        segments = list(set(segments))
        subset_ts = select_segments_subset(ts=deepcopy(ts), segments=segments)
        df = ts.to_pandas()
        subset_df = subset_ts.to_pandas()

        # fitting
        transform.fit(df)

        # transform full
        transformed_df = transform.transform(df)
        inverse_transformed_df = transform.inverse_transform(transformed_df)

        # transform subset of segments
        transformed_subset_df = transform.transform(subset_df)
        inverse_transformed_subset_df = transform.inverse_transform(transformed_subset_df)

        # checking
        assert_frame_equal(inverse_transformed_subset_df, inverse_transformed_df.loc[:, pd.IndexSlice[segments, :]])

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
    def test_inverse_transform_train_subset_segments(self, transform, dataset_name, request):
        ts = request.getfixturevalue(dataset_name)
        self._test_inverse_transform_train_subset_segments(ts, transform, segments=["segment_2"])


class TestInverseTransformFutureSubsetSegments:
    """Test inverse transform on future part of subset of segments.

    Expected that inverse transformation on subset of segments match subset of inverse transformation on full dataset.
    """

    def _test_inverse_transform_future_subset_segments(self, ts, transform, segments, horizon=7):
        # select subset of tsdataset
        subset_ts = select_segments_subset(ts=deepcopy(ts), segments=segments)
        train_df = ts.to_pandas()
        ts.transforms = [transform]
        subset_ts.transforms = [transform]

        # fitting
        transform.fit(train_df)

        # transform full
        transformed_future_ts = ts.make_future(future_steps=horizon)
        transformed_future_df = transformed_future_ts.to_pandas()
        inverse_transformed_future_ts = transform.inverse_transform(transformed_future_df)

        # transform subset of segments
        transformed_subset_future_ts = subset_ts.make_future(future_steps=horizon)
        transformed_subset_future_df = transformed_subset_future_ts.to_pandas()
        inverse_transformed_subset_future_df = transform.inverse_transform(transformed_subset_future_df)

        # checking
        assert_frame_equal(
            inverse_transformed_subset_future_df, inverse_transformed_future_ts.loc[:, pd.IndexSlice[segments, :]]
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
            (AddConstTransform(in_column="positive", value=1, inplace=True), "ts_with_exog"),
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
            (
                LambdaTransform(
                    in_column="positive",
                    transform_func=lambda x: x + 1,
                    inverse_transform_func=lambda x: x - 1,
                    inplace=True,
                ),
                "ts_with_exog",
            ),
            (LogTransform(in_column="target", inplace=False), "positive_ts"),
            (LogTransform(in_column="target", inplace=True), "positive_ts"),
            (LogTransform(in_column="positive", inplace=True), "ts_with_exog"),
            (DifferencingTransform(in_column="target", inplace=False), "regular_ts"),
            (DifferencingTransform(in_column="positive", inplace=True), "ts_with_exog"),
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
            (BoxCoxTransform(in_column="positive", mode="per-segment", inplace=True), "ts_with_exog"),
            (BoxCoxTransform(in_column="target", mode="macro", inplace=False), "positive_ts"),
            (BoxCoxTransform(in_column="target", mode="macro", inplace=True), "positive_ts"),
            (BoxCoxTransform(in_column="positive", mode="macro", inplace=True), "ts_with_exog"),
            (MaxAbsScalerTransform(in_column="target", mode="per-segment", inplace=False), "regular_ts"),
            (MaxAbsScalerTransform(in_column="target", mode="per-segment", inplace=True), "regular_ts"),
            (MaxAbsScalerTransform(in_column="positive", mode="per-segment", inplace=True), "ts_with_exog"),
            (MaxAbsScalerTransform(in_column="target", mode="macro", inplace=False), "regular_ts"),
            (MaxAbsScalerTransform(in_column="target", mode="macro", inplace=True), "regular_ts"),
            (MaxAbsScalerTransform(in_column="positive", mode="macro", inplace=True), "ts_with_exog"),
            (MinMaxScalerTransform(in_column="target", mode="per-segment", inplace=False), "regular_ts"),
            (MinMaxScalerTransform(in_column="target", mode="per-segment", inplace=True), "regular_ts"),
            (MinMaxScalerTransform(in_column="positive", mode="per-segment", inplace=True), "ts_with_exog"),
            (MinMaxScalerTransform(in_column="target", mode="macro", inplace=False), "regular_ts"),
            (MinMaxScalerTransform(in_column="target", mode="macro", inplace=True), "regular_ts"),
            (MinMaxScalerTransform(in_column="positive", mode="macro", inplace=True), "ts_with_exog"),
            (RobustScalerTransform(in_column="target", mode="per-segment", inplace=False), "regular_ts"),
            (RobustScalerTransform(in_column="target", mode="per-segment", inplace=True), "regular_ts"),
            (RobustScalerTransform(in_column="positive", mode="per-segment", inplace=True), "ts_with_exog"),
            (RobustScalerTransform(in_column="target", mode="macro", inplace=False), "regular_ts"),
            (RobustScalerTransform(in_column="target", mode="macro", inplace=True), "regular_ts"),
            (RobustScalerTransform(in_column="positive", mode="macro", inplace=True), "ts_with_exog"),
            (StandardScalerTransform(in_column="target", mode="per-segment", inplace=False), "regular_ts"),
            (StandardScalerTransform(in_column="target", mode="per-segment", inplace=True), "regular_ts"),
            (StandardScalerTransform(in_column="positive", mode="per-segment", inplace=True), "ts_with_exog"),
            (StandardScalerTransform(in_column="target", mode="macro", inplace=False), "regular_ts"),
            (StandardScalerTransform(in_column="target", mode="macro", inplace=True), "regular_ts"),
            (StandardScalerTransform(in_column="positive", mode="macro", inplace=True), "ts_with_exog"),
            (YeoJohnsonTransform(in_column="target", mode="per-segment", inplace=False), "regular_ts"),
            (YeoJohnsonTransform(in_column="target", mode="per-segment", inplace=True), "regular_ts"),
            (YeoJohnsonTransform(in_column="positive", mode="per-segment", inplace=True), "ts_with_exog"),
            (YeoJohnsonTransform(in_column="target", mode="macro", inplace=False), "regular_ts"),
            (YeoJohnsonTransform(in_column="target", mode="macro", inplace=True), "regular_ts"),
            (YeoJohnsonTransform(in_column="positive", mode="macro", inplace=True), "ts_with_exog"),
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
    def test_inverse_transform_future_subset_segments(self, transform, dataset_name, request):
        ts = request.getfixturevalue(dataset_name)
        self._test_inverse_transform_future_subset_segments(ts, transform, segments=["segment_2"])

    @to_be_fixed(ValueError, match="There should be no NaNs inside the segments")
    @pytest.mark.parametrize(
        "transform, dataset_name",
        [
            (DifferencingTransform(in_column="target", inplace=True), "regular_ts"),
            (DifferencingTransform(in_column="positive", inplace=True), "ts_with_exog"),
        ],
    )
    def test_inverse_transform_difference_fail(self, transform, dataset_name, request):
        ts = request.getfixturevalue(dataset_name)
        self._test_inverse_transform_future_subset_segments(ts, transform, segments=["segment_2"])
