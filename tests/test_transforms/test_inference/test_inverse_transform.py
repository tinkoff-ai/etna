from copy import deepcopy

import pandas as pd
import pytest
from pandas.util.testing import assert_frame_equal
from ruptures import Binseg
from sklearn.tree import DecisionTreeRegressor

from etna.analysis import StatisticsRelevanceTable
from etna.models import ProphetModel
from etna.transforms import AddConstTransform
from etna.transforms import BoxCoxTransform
from etna.transforms import ChangePointsLevelTransform
from etna.transforms import ChangePointsSegmentationTransform
from etna.transforms import ChangePointsTrendTransform
from etna.transforms import DateFlagsTransform
from etna.transforms import DensityOutliersTransform
from etna.transforms import DeseasonalityTransform
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
from tests.test_transforms.test_inference.common import find_columns_diff
from tests.utils import select_segments_subset
from tests.utils import to_be_fixed


class TestInverseTransformTrainSubsetSegments:
    """Test inverse transform on train part of subset of segments.

    Expected that inverse transformation on subset of segments match subset of inverse transformation on full dataset.
    """

    def _test_inverse_transform_train_subset_segments(self, ts, transform, segments):
        # prepare data
        segments = list(set(segments))
        subset_ts = select_segments_subset(ts=ts, segments=segments)

        # fit
        transform.fit(ts)

        # transform full
        transformed_ts = transform.transform(ts)
        inverse_transformed_df = transform.inverse_transform(transformed_ts).to_pandas()

        # transform subset of segments
        transformed_subset_ts = transform.transform(subset_ts)
        inverse_transformed_subset_df = transform.inverse_transform(transformed_subset_ts).to_pandas()

        # check
        assert_frame_equal(inverse_transformed_subset_df, inverse_transformed_df.loc[:, pd.IndexSlice[segments, :]])

    @pytest.mark.parametrize(
        "transform, dataset_name",
        [
            # decomposition
            (
                ChangePointsSegmentationTransform(
                    in_column="target",
                    change_points_model=RupturesChangePointsModel(change_points_model=Binseg(), n_bkps=5),
                ),
                "regular_ts",
            ),
            (
                ChangePointsTrendTransform(
                    in_column="target",
                ),
                "regular_ts",
            ),
            (
                ChangePointsLevelTransform(
                    in_column="target",
                ),
                "regular_ts",
            ),
            (LinearTrendTransform(in_column="target"), "regular_ts"),
            (TheilSenTrendTransform(in_column="target"), "regular_ts"),
            (STLTransform(in_column="target", period=7), "regular_ts"),
            (DeseasonalityTransform(in_column="target", period=7), "regular_ts"),
            (
                TrendTransform(
                    in_column="target",
                    change_points_model=RupturesChangePointsModel(change_points_model=Binseg(), n_bkps=5),
                ),
                "regular_ts",
            ),
            # encoders
            (LabelEncoderTransform(in_column="weekday"), "ts_with_exog"),
            (OneHotEncoderTransform(in_column="weekday"), "ts_with_exog"),
            (MeanSegmentEncoderTransform(), "regular_ts"),
            (SegmentEncoderTransform(), "regular_ts"),
            # feature_selection
            (FilterFeaturesTransform(exclude=["year"]), "ts_with_exog"),
            (GaleShapleyFeatureSelectionTransform(relevance_table=StatisticsRelevanceTable(), top_k=2), "ts_with_exog"),
            (
                MRMRFeatureSelectionTransform(
                    relevance_table=StatisticsRelevanceTable(), top_k=2, fast_redundancy=True
                ),
                "ts_with_exog",
            ),
            (
                MRMRFeatureSelectionTransform(
                    relevance_table=StatisticsRelevanceTable(),
                    top_k=2,
                    fast_redundancy=GaleShapleyFeatureSelectionTransform,
                ),
                "ts_with_exog",
            ),
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
            (HolidayTransform(mode="binary"), "regular_ts"),
            (HolidayTransform(mode="category"), "regular_ts"),
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
        # prepare data
        subset_ts = select_segments_subset(ts=ts, segments=segments)

        # fit
        transform.fit(ts)

        # transform full
        transformed_future_ts = ts.make_future(future_steps=horizon, transforms=[transform])
        inverse_transformed_future_df = transform.inverse_transform(transformed_future_ts).to_pandas()

        # transform subset of segments
        transformed_subset_future_ts = subset_ts.make_future(future_steps=horizon, transforms=[transform])
        inverse_transformed_subset_future_df = transform.inverse_transform(transformed_subset_future_ts).to_pandas()

        # check
        assert_frame_equal(
            inverse_transformed_subset_future_df, inverse_transformed_future_df.loc[:, pd.IndexSlice[segments, :]]
        )

    @pytest.mark.parametrize(
        "transform, dataset_name",
        [
            # decomposition
            (
                ChangePointsSegmentationTransform(
                    in_column="target",
                    change_points_model=RupturesChangePointsModel(change_points_model=Binseg(), n_bkps=5),
                ),
                "regular_ts",
            ),
            (
                ChangePointsTrendTransform(
                    in_column="target",
                ),
                "regular_ts",
            ),
            (
                ChangePointsTrendTransform(
                    in_column="positive",
                ),
                "ts_with_exog",
            ),
            (
                ChangePointsLevelTransform(
                    in_column="target",
                ),
                "regular_ts",
            ),
            (
                ChangePointsLevelTransform(
                    in_column="positive",
                ),
                "ts_with_exog",
            ),
            (LinearTrendTransform(in_column="target"), "regular_ts"),
            (LinearTrendTransform(in_column="positive"), "ts_with_exog"),
            (TheilSenTrendTransform(in_column="target"), "regular_ts"),
            (TheilSenTrendTransform(in_column="positive"), "ts_with_exog"),
            (STLTransform(in_column="target", period=7), "regular_ts"),
            (STLTransform(in_column="positive", period=7), "ts_with_exog"),
            (DeseasonalityTransform(in_column="target", period=7), "regular_ts"),
            (DeseasonalityTransform(in_column="positive", period=7), "ts_with_exog"),
            (
                TrendTransform(
                    in_column="target",
                    change_points_model=RupturesChangePointsModel(change_points_model=Binseg(), n_bkps=5),
                ),
                "regular_ts",
            ),
            # encoders
            (LabelEncoderTransform(in_column="weekday"), "ts_with_exog"),
            (OneHotEncoderTransform(in_column="weekday"), "ts_with_exog"),
            (MeanSegmentEncoderTransform(), "regular_ts"),
            (SegmentEncoderTransform(), "regular_ts"),
            # feature_selection
            (FilterFeaturesTransform(exclude=["year"]), "ts_with_exog"),
            (GaleShapleyFeatureSelectionTransform(relevance_table=StatisticsRelevanceTable(), top_k=2), "ts_with_exog"),
            (
                MRMRFeatureSelectionTransform(
                    relevance_table=StatisticsRelevanceTable(), top_k=2, fast_redundancy=True
                ),
                "ts_with_exog",
            ),
            (
                MRMRFeatureSelectionTransform(
                    relevance_table=StatisticsRelevanceTable(), top_k=2, fast_redundancy=False
                ),
                "ts_with_exog",
            ),
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
            (DifferencingTransform(in_column="target", inplace=True), "regular_ts"),
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
            (HolidayTransform(mode="binary"), "regular_ts"),
            (HolidayTransform(mode="category"), "regular_ts"),
            (SpecialDaysTransform(), "regular_ts"),
            (TimeFlagsTransform(), "regular_ts"),
        ],
    )
    def test_inverse_transform_future_subset_segments(self, transform, dataset_name, request):
        ts = request.getfixturevalue(dataset_name)
        self._test_inverse_transform_future_subset_segments(ts, transform, segments=["segment_2"])


class TestInverseTransformTrainNewSegments:
    """Test inverse transform on train part of new segments.

    Expected that inverse transformation creates columns, removes columns and reverts values back to original.
    """

    def _test_inverse_transform_train_new_segments(self, ts, transform, train_segments, expected_changes):
        # prepare data
        train_segments = list(set(train_segments))
        forecast_segments = list(set(ts.segments) - set(train_segments))
        train_ts = select_segments_subset(ts=ts, segments=train_segments)
        test_ts = select_segments_subset(ts=ts, segments=forecast_segments)

        # fit
        transform.fit(train_ts)

        # transform
        transformed_test_ts = transform.transform(deepcopy(test_ts))

        # inverse transform
        inverse_transformed_test_ts = transform.inverse_transform(deepcopy(transformed_test_ts))

        # check
        expected_columns_to_create = expected_changes.get("create", set())
        expected_columns_to_remove = expected_changes.get("remove", set())
        expected_columns_to_change = expected_changes.get("change", set())
        flat_test_df = test_ts.to_pandas(flatten=True)
        flat_transformed_test_df = transformed_test_ts.to_pandas(flatten=True)
        flat_inverse_transformed_test_df = inverse_transformed_test_ts.to_pandas(flatten=True)
        created_columns, removed_columns, changed_columns = find_columns_diff(
            flat_transformed_test_df, flat_inverse_transformed_test_df
        )

        assert created_columns == expected_columns_to_create
        assert removed_columns == expected_columns_to_remove
        assert changed_columns == expected_columns_to_change
        pd.testing.assert_frame_equal(flat_test_df[changed_columns], flat_inverse_transformed_test_df[changed_columns])

    @pytest.mark.parametrize(
        "transform, dataset_name, expected_changes",
        [
            # encoders
            (LabelEncoderTransform(in_column="weekday", out_column="res"), "ts_with_exog", {}),
            (
                OneHotEncoderTransform(in_column="weekday", out_column="res"),
                "ts_with_exog",
                {},
            ),
            # feature_selection
            (FilterFeaturesTransform(exclude=["year"]), "ts_with_exog", {}),
            (FilterFeaturesTransform(exclude=["year"], return_features=True), "ts_with_exog", {"create": {"year"}}),
            (
                GaleShapleyFeatureSelectionTransform(relevance_table=StatisticsRelevanceTable(), top_k=2),
                "ts_with_exog",
                {},
            ),
            (
                GaleShapleyFeatureSelectionTransform(
                    relevance_table=StatisticsRelevanceTable(), top_k=2, return_features=True
                ),
                "ts_with_exog",
                {"create": {"year", "weekday", "month"}},
            ),
            (
                MRMRFeatureSelectionTransform(
                    relevance_table=StatisticsRelevanceTable(), top_k=2, fast_redundancy=True
                ),
                "ts_with_exog",
                {},
            ),
            (
                MRMRFeatureSelectionTransform(
                    relevance_table=StatisticsRelevanceTable(), top_k=2, fast_redundancy=False
                ),
                "ts_with_exog",
                {},
            ),
            (
                MRMRFeatureSelectionTransform(
                    relevance_table=StatisticsRelevanceTable(), top_k=2, return_features=True, fast_redundancy=True
                ),
                "ts_with_exog",
                {"create": {"monthday", "positive", "weekday"}},
            ),
            (
                MRMRFeatureSelectionTransform(
                    relevance_table=StatisticsRelevanceTable(), top_k=2, return_features=True, fast_redundancy=False
                ),
                "ts_with_exog",
                {"create": {"monthday", "positive", "weekday"}},
            ),
            (
                TreeFeatureSelectionTransform(model=DecisionTreeRegressor(random_state=42), top_k=2),
                "ts_with_exog",
                {},
            ),
            (
                TreeFeatureSelectionTransform(
                    model=DecisionTreeRegressor(random_state=42), top_k=2, return_features=True
                ),
                "ts_with_exog",
                {"create": {"month", "year", "weekday"}},
            ),
            # math
            (
                AddConstTransform(in_column="target", value=1, inplace=False, out_column="res"),
                "regular_ts",
                {},
            ),
            (AddConstTransform(in_column="target", value=1, inplace=True), "regular_ts", {"change": {"target"}}),
            (
                LagTransform(in_column="target", lags=[1, 2, 3], out_column="res"),
                "regular_ts",
                {},
            ),
            (
                LambdaTransform(in_column="target", transform_func=lambda x: x + 1, inplace=False, out_column="res"),
                "regular_ts",
                {},
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
            (LogTransform(in_column="target", inplace=False, out_column="res"), "positive_ts", {}),
            (LogTransform(in_column="target", inplace=True), "positive_ts", {"change": {"target"}}),
            (
                DifferencingTransform(in_column="target", inplace=False, out_column="res"),
                "regular_ts",
                {},
            ),
            (MADTransform(in_column="target", window=7, out_column="res"), "regular_ts", {}),
            (MaxTransform(in_column="target", window=7, out_column="res"), "regular_ts", {}),
            (MeanTransform(in_column="target", window=7, out_column="res"), "regular_ts", {}),
            (MedianTransform(in_column="target", window=7, out_column="res"), "regular_ts", {}),
            (
                MinMaxDifferenceTransform(in_column="target", window=7, out_column="res"),
                "regular_ts",
                {},
            ),
            (MinTransform(in_column="target", window=7, out_column="res"), "regular_ts", {}),
            (
                QuantileTransform(in_column="target", quantile=0.9, window=7, out_column="res"),
                "regular_ts",
                {},
            ),
            (StdTransform(in_column="target", window=7, out_column="res"), "regular_ts", {}),
            (SumTransform(in_column="target", window=7, out_column="res"), "regular_ts", {}),
            (
                BoxCoxTransform(in_column="target", mode="macro", inplace=False, out_column="res"),
                "positive_ts",
                {},
            ),
            (BoxCoxTransform(in_column="target", mode="macro", inplace=True), "positive_ts", {"change": {"target"}}),
            (
                MaxAbsScalerTransform(in_column="target", mode="macro", inplace=False, out_column="res"),
                "regular_ts",
                {},
            ),
            (
                MaxAbsScalerTransform(in_column="target", mode="macro", inplace=True),
                "regular_ts",
                {"change": {"target"}},
            ),
            (
                MinMaxScalerTransform(in_column="target", mode="macro", inplace=False, out_column="res"),
                "regular_ts",
                {},
            ),
            (
                MinMaxScalerTransform(in_column="target", mode="macro", inplace=True),
                "regular_ts",
                {"change": {"target"}},
            ),
            (
                RobustScalerTransform(in_column="target", mode="macro", inplace=False, out_column="res"),
                "regular_ts",
                {},
            ),
            (
                RobustScalerTransform(in_column="target", mode="macro", inplace=True),
                "regular_ts",
                {"change": {"target"}},
            ),
            (
                StandardScalerTransform(in_column="target", mode="macro", inplace=False, out_column="res"),
                "regular_ts",
                {},
            ),
            (
                StandardScalerTransform(in_column="target", mode="macro", inplace=True),
                "regular_ts",
                {"change": {"target"}},
            ),
            (
                YeoJohnsonTransform(in_column="target", mode="macro", inplace=False, out_column="res"),
                "regular_ts",
                {},
            ),
            (YeoJohnsonTransform(in_column="target", mode="macro", inplace=True), "regular_ts", {"change": {"target"}}),
            # timestamp
            (
                DateFlagsTransform(out_column="res"),
                "regular_ts",
                {},
            ),
            (
                FourierTransform(period=7, order=2, out_column="res"),
                "regular_ts",
                {},
            ),
            (HolidayTransform(out_column="res", mode="binary"), "regular_ts", {}),
            (HolidayTransform(out_column="res", mode="category"), "regular_ts", {}),
            (
                TimeFlagsTransform(out_column="res"),
                "regular_ts",
                {},
            ),
        ],
    )
    def test_inverse_transform_train_new_segments(self, transform, dataset_name, expected_changes, request):
        ts = request.getfixturevalue(dataset_name)
        self._test_inverse_transform_train_new_segments(
            ts, transform, train_segments=["segment_1", "segment_2"], expected_changes=expected_changes
        )

    @pytest.mark.parametrize(
        "transform, dataset_name",
        [
            # decomposition
            (
                ChangePointsSegmentationTransform(
                    in_column="target",
                    change_points_model=RupturesChangePointsModel(change_points_model=Binseg(), n_bkps=5),
                ),
                "regular_ts",
            ),
            (
                ChangePointsTrendTransform(in_column="target"),
                "regular_ts",
            ),
            (
                ChangePointsLevelTransform(in_column="target"),
                "regular_ts",
            ),
            (LinearTrendTransform(in_column="target"), "regular_ts"),
            (TheilSenTrendTransform(in_column="target"), "regular_ts"),
            (STLTransform(in_column="target", period=7), "regular_ts"),
            (DeseasonalityTransform(in_column="target", period=7), "regular_ts"),
            (
                TrendTransform(
                    in_column="target",
                    change_points_model=RupturesChangePointsModel(change_points_model=Binseg(), n_bkps=5),
                ),
                "regular_ts",
            ),
            # encoders
            (MeanSegmentEncoderTransform(), "regular_ts"),
            (SegmentEncoderTransform(), "regular_ts"),
            # math
            (DifferencingTransform(in_column="target", inplace=True), "regular_ts"),
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
            # outliers
            (DensityOutliersTransform(in_column="target"), "ts_with_outliers"),
            (MedianOutliersTransform(in_column="target"), "ts_with_outliers"),
            (PredictionIntervalOutliersTransform(in_column="target", model=ProphetModel), "ts_with_outliers"),
            # timestamp
            (SpecialDaysTransform(), "regular_ts"),
        ],
    )
    def test_inverse_transform_train_new_segments_not_implemented(self, transform, dataset_name, request):
        ts = request.getfixturevalue(dataset_name)
        with pytest.raises(NotImplementedError):
            self._test_inverse_transform_train_new_segments(
                ts, transform, train_segments=["segment_1", "segment_2"], expected_changes={}
            )


class TestInverseTransformFutureNewSegments:
    """Test inverse transform on future part of new segments.

    Expected that inverse transformation creates columns, removes columns and reverts values back to original.
    """

    def _test_inverse_transform_future_new_segments(self, ts, transform, train_segments, expected_changes, horizon=7):
        # prepare data
        train_segments = list(set(train_segments))
        forecast_segments = list(set(ts.segments) - set(train_segments))
        train_ts = select_segments_subset(ts=ts, segments=train_segments)
        new_segments_ts = select_segments_subset(ts=ts, segments=forecast_segments)

        # fit
        transform.fit(train_ts)

        # prepare ts without transform
        test_ts = new_segments_ts.make_future(future_steps=horizon)

        # transform
        transformed_test_ts = new_segments_ts.make_future(future_steps=horizon, transforms=[transform])

        # inverse transform
        inverse_transformed_test_ts = transform.inverse_transform(deepcopy(transformed_test_ts))

        # check
        expected_columns_to_create = expected_changes.get("create", set())
        expected_columns_to_remove = expected_changes.get("remove", set())
        expected_columns_to_change = expected_changes.get("change", set())
        flat_test_df = test_ts.to_pandas(flatten=True)
        flat_transformed_test_df = transformed_test_ts.to_pandas(flatten=True)
        flat_inverse_transformed_test_df = inverse_transformed_test_ts.to_pandas(flatten=True)
        created_columns, removed_columns, changed_columns = find_columns_diff(
            flat_transformed_test_df, flat_inverse_transformed_test_df
        )

        assert created_columns == expected_columns_to_create
        assert removed_columns == expected_columns_to_remove
        assert changed_columns == expected_columns_to_change
        pd.testing.assert_frame_equal(flat_test_df[changed_columns], flat_inverse_transformed_test_df[changed_columns])

    @pytest.mark.parametrize(
        "transform, dataset_name, expected_changes",
        [
            # encoders
            (LabelEncoderTransform(in_column="weekday", out_column="res"), "ts_with_exog", {}),
            (
                OneHotEncoderTransform(in_column="weekday", out_column="res"),
                "ts_with_exog",
                {},
            ),
            # feature_selection
            (FilterFeaturesTransform(exclude=["year"]), "ts_with_exog", {}),
            (
                GaleShapleyFeatureSelectionTransform(relevance_table=StatisticsRelevanceTable(), top_k=2),
                "ts_with_exog",
                {},
            ),
            (
                MRMRFeatureSelectionTransform(
                    relevance_table=StatisticsRelevanceTable(), top_k=2, fast_redundancy=True
                ),
                "ts_with_exog",
                {},
            ),
            (
                MRMRFeatureSelectionTransform(
                    relevance_table=StatisticsRelevanceTable(), top_k=2, fast_redundancy=False
                ),
                "ts_with_exog",
                {},
            ),
            (
                TreeFeatureSelectionTransform(model=DecisionTreeRegressor(random_state=42), top_k=2),
                "ts_with_exog",
                {},
            ),
            # math
            (
                AddConstTransform(in_column="target", value=1, inplace=False, out_column="res"),
                "regular_ts",
                {},
            ),
            (AddConstTransform(in_column="target", value=1, inplace=True), "regular_ts", {}),
            (AddConstTransform(in_column="positive", value=1, inplace=True), "ts_with_exog", {"change": {"positive"}}),
            (
                LagTransform(in_column="target", lags=[1, 2, 3], out_column="res"),
                "regular_ts",
                {},
            ),
            (
                LambdaTransform(in_column="target", transform_func=lambda x: x + 1, inplace=False, out_column="res"),
                "regular_ts",
                {},
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
            (
                LambdaTransform(
                    in_column="positive",
                    transform_func=lambda x: x + 1,
                    inverse_transform_func=lambda x: x - 1,
                    inplace=True,
                ),
                "ts_with_exog",
                {"change": {"positive"}},
            ),
            (LogTransform(in_column="target", inplace=False, out_column="res"), "positive_ts", {}),
            (LogTransform(in_column="target", inplace=True), "positive_ts", {}),
            (LogTransform(in_column="positive", inplace=True), "ts_with_exog", {"change": {"positive"}}),
            (
                DifferencingTransform(in_column="target", inplace=False, out_column="res"),
                "regular_ts",
                {},
            ),
            (MADTransform(in_column="target", window=14, out_column="res"), "regular_ts", {}),
            (MaxTransform(in_column="target", window=14, out_column="res"), "regular_ts", {}),
            (MeanTransform(in_column="target", window=14, out_column="res"), "regular_ts", {}),
            (MedianTransform(in_column="target", window=14, out_column="res"), "regular_ts", {}),
            (
                MinMaxDifferenceTransform(in_column="target", window=14, out_column="res"),
                "regular_ts",
                {},
            ),
            (MinTransform(in_column="target", window=14, out_column="res"), "regular_ts", {}),
            (
                QuantileTransform(in_column="target", quantile=0.9, window=14, out_column="res"),
                "regular_ts",
                {},
            ),
            (StdTransform(in_column="target", window=14, out_column="res"), "regular_ts", {}),
            (SumTransform(in_column="target", window=14, out_column="res"), "regular_ts", {}),
            (
                BoxCoxTransform(in_column="target", mode="macro", inplace=False, out_column="res"),
                "positive_ts",
                {},
            ),
            (BoxCoxTransform(in_column="target", mode="macro", inplace=True), "positive_ts", {}),
            (
                BoxCoxTransform(in_column="positive", mode="macro", inplace=True),
                "ts_with_exog",
                {"change": {"positive"}},
            ),
            (
                MaxAbsScalerTransform(in_column="target", mode="macro", inplace=False, out_column="res"),
                "regular_ts",
                {},
            ),
            (
                MaxAbsScalerTransform(in_column="target", mode="macro", inplace=True),
                "regular_ts",
                {},
            ),
            (
                MaxAbsScalerTransform(in_column="positive", mode="macro", inplace=True),
                "ts_with_exog",
                {"change": {"positive"}},
            ),
            (
                MinMaxScalerTransform(in_column="target", mode="macro", inplace=False, out_column="res"),
                "regular_ts",
                {},
            ),
            (
                MinMaxScalerTransform(in_column="target", mode="macro", inplace=True),
                "regular_ts",
                {},
            ),
            (
                MinMaxScalerTransform(in_column="positive", mode="macro", inplace=True),
                "ts_with_exog",
                {"change": {"positive"}},
            ),
            (
                RobustScalerTransform(in_column="target", mode="macro", inplace=False, out_column="res"),
                "regular_ts",
                {},
            ),
            (
                RobustScalerTransform(in_column="target", mode="macro", inplace=True),
                "regular_ts",
                {},
            ),
            (
                RobustScalerTransform(in_column="positive", mode="macro", inplace=True),
                "ts_with_exog",
                {"change": {"positive"}},
            ),
            (
                StandardScalerTransform(in_column="target", mode="macro", inplace=False, out_column="res"),
                "regular_ts",
                {},
            ),
            (
                StandardScalerTransform(in_column="target", mode="macro", inplace=True),
                "regular_ts",
                {},
            ),
            (
                StandardScalerTransform(in_column="positive", mode="macro", inplace=True),
                "ts_with_exog",
                {"change": {"positive"}},
            ),
            (
                YeoJohnsonTransform(in_column="target", mode="macro", inplace=False, out_column="res"),
                "regular_ts",
                {},
            ),
            (YeoJohnsonTransform(in_column="target", mode="macro", inplace=True), "regular_ts", {}),
            (
                YeoJohnsonTransform(in_column="positive", mode="macro", inplace=True),
                "ts_with_exog",
                {"change": {"positive"}},
            ),
            # timestamp
            (
                DateFlagsTransform(out_column="res"),
                "regular_ts",
                {},
            ),
            (
                FourierTransform(period=7, order=2, out_column="res"),
                "regular_ts",
                {},
            ),
            (HolidayTransform(out_column="res", mode="binary"), "regular_ts", {}),
            (HolidayTransform(out_column="res", mode="category"), "regular_ts", {}),
            (
                TimeFlagsTransform(out_column="res"),
                "regular_ts",
                {},
            ),
        ],
    )
    def test_inverse_transform_future_new_segments(self, transform, dataset_name, expected_changes, request):
        ts = request.getfixturevalue(dataset_name)
        self._test_inverse_transform_future_new_segments(
            ts, transform, train_segments=["segment_1", "segment_2"], expected_changes=expected_changes
        )

    @pytest.mark.parametrize(
        "transform, dataset_name",
        [
            # decomposition
            (
                ChangePointsSegmentationTransform(
                    in_column="target",
                    change_points_model=RupturesChangePointsModel(change_points_model=Binseg(), n_bkps=5),
                ),
                "regular_ts",
            ),
            (
                ChangePointsTrendTransform(in_column="target"),
                "regular_ts",
            ),
            (
                ChangePointsLevelTransform(in_column="target"),
                "regular_ts",
            ),
            (LinearTrendTransform(in_column="target"), "regular_ts"),
            (TheilSenTrendTransform(in_column="target"), "regular_ts"),
            (STLTransform(in_column="target", period=7), "regular_ts"),
            (DeseasonalityTransform(in_column="target", period=7), "regular_ts"),
            (
                TrendTransform(
                    in_column="target",
                    change_points_model=RupturesChangePointsModel(change_points_model=Binseg(), n_bkps=5),
                ),
                "regular_ts",
            ),
            # encoders
            (MeanSegmentEncoderTransform(), "regular_ts"),
            (SegmentEncoderTransform(), "regular_ts"),
            # math
            (DifferencingTransform(in_column="target", inplace=True), "regular_ts"),
            (DifferencingTransform(in_column="positive", inplace=True), "ts_with_exog"),
            (BoxCoxTransform(in_column="target", mode="per-segment", inplace=False), "positive_ts"),
            (BoxCoxTransform(in_column="target", mode="per-segment", inplace=True), "positive_ts"),
            (BoxCoxTransform(in_column="positive", mode="per-segment", inplace=True), "ts_with_exog"),
            (MaxAbsScalerTransform(in_column="target", mode="per-segment", inplace=False), "regular_ts"),
            (MaxAbsScalerTransform(in_column="target", mode="per-segment", inplace=True), "regular_ts"),
            (MaxAbsScalerTransform(in_column="positive", mode="per-segment", inplace=True), "ts_with_exog"),
            (MinMaxScalerTransform(in_column="target", mode="per-segment", inplace=False), "regular_ts"),
            (MinMaxScalerTransform(in_column="target", mode="per-segment", inplace=True), "regular_ts"),
            (MinMaxScalerTransform(in_column="positive", mode="per-segment", inplace=True), "ts_with_exog"),
            (RobustScalerTransform(in_column="target", mode="per-segment", inplace=False), "regular_ts"),
            (RobustScalerTransform(in_column="target", mode="per-segment", inplace=True), "regular_ts"),
            (RobustScalerTransform(in_column="positive", mode="per-segment", inplace=True), "ts_with_exog"),
            (StandardScalerTransform(in_column="target", mode="per-segment", inplace=False), "regular_ts"),
            (StandardScalerTransform(in_column="target", mode="per-segment", inplace=True), "regular_ts"),
            (StandardScalerTransform(in_column="positive", mode="per-segment", inplace=True), "ts_with_exog"),
            (YeoJohnsonTransform(in_column="target", mode="per-segment", inplace=False), "regular_ts"),
            (YeoJohnsonTransform(in_column="target", mode="per-segment", inplace=True), "regular_ts"),
            (YeoJohnsonTransform(in_column="positive", mode="per-segment", inplace=True), "ts_with_exog"),
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
            # outliers
            (DensityOutliersTransform(in_column="target"), "ts_with_outliers"),
            (MedianOutliersTransform(in_column="target"), "ts_with_outliers"),
            (PredictionIntervalOutliersTransform(in_column="target", model=ProphetModel), "ts_with_outliers"),
            # timestamp
            (SpecialDaysTransform(), "regular_ts"),
        ],
    )
    def test_inverse_transform_future_new_segments_not_implemented(self, transform, dataset_name, request):
        ts = request.getfixturevalue(dataset_name)
        with pytest.raises(NotImplementedError):
            self._test_inverse_transform_future_new_segments(
                ts, transform, train_segments=["segment_1", "segment_2"], expected_changes={}
            )

    @to_be_fixed(raises=Exception)
    @pytest.mark.parametrize(
        "transform, dataset_name, expected_changes",
        [
            # feature_selection
            # TODO: working incorrectly, should fail
            (FilterFeaturesTransform(exclude=["year"], return_features=True), "ts_with_exog", {"create": {"year"}}),
            (
                GaleShapleyFeatureSelectionTransform(
                    relevance_table=StatisticsRelevanceTable(), top_k=2, return_features=True
                ),
                "ts_with_exog",
                {"create": {"monthday", "positive", "weekday", "year", "month"}},
            ),
            (
                MRMRFeatureSelectionTransform(
                    relevance_table=StatisticsRelevanceTable(), top_k=2, return_features=True, fast_redundancy=True
                ),
                "ts_with_exog",
                {"create": {"positive", "monthday", "weekday"}},
            ),
            (
                MRMRFeatureSelectionTransform(
                    relevance_table=StatisticsRelevanceTable(), top_k=2, return_features=True, fast_redundancy=False
                ),
                "ts_with_exog",
                {"create": {"positive", "monthday", "weekday"}},
            ),
            (
                TreeFeatureSelectionTransform(
                    model=DecisionTreeRegressor(random_state=42), top_k=2, return_features=True
                ),
                "ts_with_exog",
                {"create": {"year", "month", "weekday"}},
            ),
        ],
    )
    def test_inverse_transform_future_new_segments_failed_error(
        self, transform, dataset_name, expected_changes, request
    ):
        ts = request.getfixturevalue(dataset_name)
        self._test_inverse_transform_future_new_segments(
            ts, transform, train_segments=["segment_1", "segment_2"], expected_changes=expected_changes
        )


class TestInverseTransformFutureWithTarget:
    """Test inverse transform on future dataset with known target.

    Expected that inverse transformation creates columns, removes columns and reverts values back to original.
    """

    def _test_inverse_transform_future_with_target(
        self, ts, transform, expected_changes, gap_size=7, transform_size=50
    ):
        # prepare data
        train_ts, future_full_ts = ts.train_test_split(test_size=gap_size + transform_size)
        _, test_ts = future_full_ts.train_test_split(test_size=transform_size)

        # fit
        transform.fit(train_ts)

        # transform
        transformed_test_ts = transform.transform(deepcopy(test_ts))

        # inverse transform
        inverse_transformed_test_ts = transform.inverse_transform(deepcopy(transformed_test_ts))

        # check
        expected_columns_to_create = expected_changes.get("create", set())
        expected_columns_to_remove = expected_changes.get("remove", set())
        expected_columns_to_change = expected_changes.get("change", set())
        flat_test_df = test_ts.to_pandas(flatten=True)
        flat_transformed_test_df = transformed_test_ts.to_pandas(flatten=True)
        flat_inverse_transformed_test_df = inverse_transformed_test_ts.to_pandas(flatten=True)
        created_columns, removed_columns, changed_columns = find_columns_diff(
            flat_transformed_test_df, flat_inverse_transformed_test_df
        )

        assert created_columns == expected_columns_to_create
        assert removed_columns == expected_columns_to_remove
        assert changed_columns == expected_columns_to_change
        pd.testing.assert_frame_equal(
            flat_test_df[changed_columns],
            flat_inverse_transformed_test_df[changed_columns],
        )

    @pytest.mark.parametrize(
        "transform, dataset_name, expected_changes",
        [
            # decomposition
            (
                ChangePointsSegmentationTransform(
                    in_column="target",
                    change_points_model=RupturesChangePointsModel(change_points_model=Binseg(), n_bkps=5),
                    out_column="res",
                ),
                "regular_ts",
                {},
            ),
            (
                ChangePointsTrendTransform(in_column="target"),
                "regular_ts",
                {"change": {"target"}},
            ),
            (
                ChangePointsLevelTransform(in_column="target"),
                "regular_ts",
                {"change": {"target"}},
            ),
            (LinearTrendTransform(in_column="target"), "regular_ts", {"change": {"target"}}),
            (TheilSenTrendTransform(in_column="target"), "regular_ts", {"change": {"target"}}),
            (STLTransform(in_column="target", period=7), "regular_ts", {"change": {"target"}}),
            (DeseasonalityTransform(in_column="target", period=7), "regular_ts", {"change": {"target"}}),
            (
                TrendTransform(
                    in_column="target",
                    change_points_model=RupturesChangePointsModel(change_points_model=Binseg(), n_bkps=5),
                    out_column="res",
                ),
                "regular_ts",
                {},
            ),
            # encoders
            (LabelEncoderTransform(in_column="weekday", out_column="res"), "ts_with_exog", {}),
            (
                OneHotEncoderTransform(in_column="weekday", out_column="res"),
                "ts_with_exog",
                {},
            ),
            (MeanSegmentEncoderTransform(), "regular_ts", {}),
            (SegmentEncoderTransform(), "regular_ts", {}),
            # feature_selection
            (FilterFeaturesTransform(exclude=["year"]), "ts_with_exog", {}),
            (FilterFeaturesTransform(exclude=["year"], return_features=True), "ts_with_exog", {"create": {"year"}}),
            (
                GaleShapleyFeatureSelectionTransform(relevance_table=StatisticsRelevanceTable(), top_k=2),
                "ts_with_exog",
                {},
            ),
            (
                GaleShapleyFeatureSelectionTransform(
                    relevance_table=StatisticsRelevanceTable(), top_k=2, return_features=True
                ),
                "ts_with_exog",
                {"create": {"month", "year", "positive"}},
            ),
            (
                MRMRFeatureSelectionTransform(
                    relevance_table=StatisticsRelevanceTable(), top_k=2, fast_redundancy=True
                ),
                "ts_with_exog",
                {},
            ),
            (
                MRMRFeatureSelectionTransform(
                    relevance_table=StatisticsRelevanceTable(), top_k=2, fast_redundancy=False
                ),
                "ts_with_exog",
                {},
            ),
            (
                MRMRFeatureSelectionTransform(
                    relevance_table=StatisticsRelevanceTable(),
                    top_k=2,
                    return_features=True,
                    fast_redundancy=True,
                ),
                "ts_with_exog",
                {"create": {"weekday", "monthday", "positive"}},
            ),
            (
                MRMRFeatureSelectionTransform(
                    relevance_table=StatisticsRelevanceTable(), top_k=2, return_features=True, fast_redundancy=False
                ),
                "ts_with_exog",
                {"create": {"weekday", "monthday", "positive"}},
            ),
            (
                TreeFeatureSelectionTransform(model=DecisionTreeRegressor(random_state=42), top_k=2),
                "ts_with_exog",
                {},
            ),
            (
                TreeFeatureSelectionTransform(
                    model=DecisionTreeRegressor(random_state=42), top_k=2, return_features=True
                ),
                "ts_with_exog",
                {"create": {"year", "month", "weekday"}},
            ),
            # math
            (
                AddConstTransform(in_column="target", value=1, inplace=False, out_column="res"),
                "regular_ts",
                {},
            ),
            (AddConstTransform(in_column="target", value=1, inplace=True), "regular_ts", {"change": {"target"}}),
            (
                LagTransform(in_column="target", lags=[1, 2, 3], out_column="res"),
                "regular_ts",
                {},
            ),
            (
                LambdaTransform(in_column="target", transform_func=lambda x: x + 1, inplace=False, out_column="res"),
                "regular_ts",
                {},
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
            (LogTransform(in_column="target", inplace=False, out_column="res"), "positive_ts", {}),
            (LogTransform(in_column="target", inplace=True), "positive_ts", {"change": {"target"}}),
            (
                DifferencingTransform(in_column="target", inplace=False, out_column="res"),
                "regular_ts",
                {},
            ),
            (MADTransform(in_column="target", window=7, out_column="res"), "regular_ts", {}),
            (MaxTransform(in_column="target", window=7, out_column="res"), "regular_ts", {}),
            (MeanTransform(in_column="target", window=7, out_column="res"), "regular_ts", {}),
            (MedianTransform(in_column="target", window=7, out_column="res"), "regular_ts", {}),
            (
                MinMaxDifferenceTransform(in_column="target", window=7, out_column="res"),
                "regular_ts",
                {},
            ),
            (MinTransform(in_column="target", window=7, out_column="res"), "regular_ts", {}),
            (
                QuantileTransform(in_column="target", quantile=0.9, window=7, out_column="res"),
                "regular_ts",
                {},
            ),
            (StdTransform(in_column="target", window=7, out_column="res"), "regular_ts", {}),
            (SumTransform(in_column="target", window=7, out_column="res"), "regular_ts", {}),
            (
                BoxCoxTransform(in_column="target", mode="per-segment", inplace=False, out_column="res"),
                "positive_ts",
                {},
            ),
            (
                BoxCoxTransform(in_column="target", mode="per-segment", inplace=True),
                "positive_ts",
                {"change": {"target"}},
            ),
            (
                BoxCoxTransform(in_column="target", mode="macro", inplace=False, out_column="res"),
                "positive_ts",
                {},
            ),
            (BoxCoxTransform(in_column="target", mode="macro", inplace=True), "positive_ts", {"change": {"target"}}),
            (
                MaxAbsScalerTransform(in_column="target", mode="per-segment", inplace=False, out_column="res"),
                "regular_ts",
                {},
            ),
            (
                MaxAbsScalerTransform(in_column="target", mode="per-segment", inplace=True),
                "regular_ts",
                {"change": {"target"}},
            ),
            (
                MaxAbsScalerTransform(in_column="target", mode="macro", inplace=False, out_column="res"),
                "regular_ts",
                {},
            ),
            (
                MaxAbsScalerTransform(in_column="target", mode="macro", inplace=True),
                "regular_ts",
                {"change": {"target"}},
            ),
            (
                MinMaxScalerTransform(in_column="target", mode="per-segment", inplace=False, out_column="res"),
                "regular_ts",
                {},
            ),
            (
                # setting clip=False is important
                MinMaxScalerTransform(in_column="target", mode="per-segment", clip=False, inplace=True),
                "regular_ts",
                {"change": {"target"}},
            ),
            (
                MinMaxScalerTransform(in_column="target", mode="macro", inplace=False, out_column="res"),
                "regular_ts",
                {},
            ),
            (
                # setting clip=False is important
                MinMaxScalerTransform(in_column="target", mode="macro", clip=False, inplace=True),
                "regular_ts",
                {"change": {"target"}},
            ),
            (
                RobustScalerTransform(in_column="target", mode="per-segment", inplace=False, out_column="res"),
                "regular_ts",
                {},
            ),
            (
                RobustScalerTransform(in_column="target", mode="per-segment", inplace=True),
                "regular_ts",
                {"change": {"target"}},
            ),
            (
                RobustScalerTransform(in_column="target", mode="macro", inplace=False, out_column="res"),
                "regular_ts",
                {},
            ),
            (
                RobustScalerTransform(in_column="target", mode="macro", inplace=True),
                "regular_ts",
                {"change": {"target"}},
            ),
            (
                StandardScalerTransform(in_column="target", mode="per-segment", inplace=False, out_column="res"),
                "regular_ts",
                {},
            ),
            (
                StandardScalerTransform(in_column="target", mode="per-segment", inplace=True),
                "regular_ts",
                {"change": {"target"}},
            ),
            (
                StandardScalerTransform(in_column="target", mode="macro", inplace=False, out_column="res"),
                "regular_ts",
                {},
            ),
            (
                StandardScalerTransform(in_column="target", mode="macro", inplace=True),
                "regular_ts",
                {"change": {"target"}},
            ),
            (
                YeoJohnsonTransform(in_column="target", mode="per-segment", inplace=False, out_column="res"),
                "regular_ts",
                {},
            ),
            (
                YeoJohnsonTransform(in_column="target", mode="per-segment", inplace=True),
                "regular_ts",
                {"change": {"target"}},
            ),
            (
                YeoJohnsonTransform(in_column="target", mode="macro", inplace=False, out_column="res"),
                "regular_ts",
                {},
            ),
            (YeoJohnsonTransform(in_column="target", mode="macro", inplace=True), "regular_ts", {"change": {"target"}}),
            # missing_values
            (
                ResampleWithDistributionTransform(
                    in_column="regressor_exog", distribution_column="target", inplace=False, out_column="res"
                ),
                "ts_to_resample",
                {},
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
                {},
            ),
            (
                FourierTransform(period=7, order=2, out_column="res"),
                "regular_ts",
                {},
            ),
            (HolidayTransform(out_column="res", mode="binary"), "regular_ts", {}),
            (HolidayTransform(out_column="res", mode="category"), "regular_ts", {}),
            (
                TimeFlagsTransform(out_column="res"),
                "regular_ts",
                {},
            ),
            (SpecialDaysTransform(), "regular_ts", {}),
        ],
    )
    def test_inverse_transform_future_with_target(self, transform, dataset_name, expected_changes, request):
        ts = request.getfixturevalue(dataset_name)
        self._test_inverse_transform_future_with_target(ts, transform, expected_changes=expected_changes)

    @pytest.mark.parametrize(
        "transform, dataset_name, expected_changes",
        [
            (DifferencingTransform(in_column="target", inplace=True), "regular_ts", {}),
        ],
    )
    def test_inverse_transform_future_with_target_fail_difference(
        self, transform, dataset_name, expected_changes, request
    ):
        ts = request.getfixturevalue(dataset_name)
        with pytest.raises(ValueError, match="Test should go after the train without gaps"):
            self._test_inverse_transform_future_with_target(ts, transform, expected_changes=expected_changes)

    # It is the only transform that doesn't change values back during `inverse_transform`
    @to_be_fixed(raises=AssertionError)
    @pytest.mark.parametrize(
        "transform, dataset_name, expected_changes",
        [
            (
                ResampleWithDistributionTransform(
                    in_column="regressor_exog", distribution_column="target", inplace=True
                ),
                "ts_to_resample",
                {"change": {"regressor_exog"}},
            ),
        ],
    )
    def test_inverse_transform_future_with_target_fail_resample(
        self, transform, dataset_name, expected_changes, request
    ):
        ts = request.getfixturevalue(dataset_name)
        self._test_inverse_transform_future_with_target(ts, transform, expected_changes=expected_changes)


class TestInverseTransformFutureWithoutTarget:
    """Test inverse transform on future dataset with unknown target.

    Expected that inverse transformation creates columns, removes columns and reverts values back to original.
    """

    def _test_inverse_transform_future_without_target(
        self, ts, transform, expected_changes, gap_size=28, transform_size=7
    ):
        # prepare data
        train_ts, future_ts = ts.train_test_split(test_size=gap_size)
        future_ts = future_ts

        # fit
        transform.fit(train_ts)

        # prepare ts without transform
        test_ts = future_ts.make_future(future_steps=transform_size)

        # transform
        transformed_test_ts = future_ts.make_future(future_steps=transform_size, transforms=[transform])

        # inverse transform
        inverse_transformed_test_ts = transform.inverse_transform(deepcopy(transformed_test_ts))

        # check
        expected_columns_to_create = expected_changes.get("create", set())
        expected_columns_to_remove = expected_changes.get("remove", set())
        expected_columns_to_change = expected_changes.get("change", set())
        flat_test_df = test_ts.to_pandas(flatten=True)
        flat_transformed_test_df = transformed_test_ts.to_pandas(flatten=True)
        flat_inverse_transformed_test_df = inverse_transformed_test_ts.to_pandas(flatten=True)
        created_columns, removed_columns, changed_columns = find_columns_diff(
            flat_transformed_test_df, flat_inverse_transformed_test_df
        )

        assert created_columns == expected_columns_to_create
        assert removed_columns == expected_columns_to_remove
        assert changed_columns == expected_columns_to_change
        pd.testing.assert_frame_equal(
            flat_test_df[changed_columns],
            flat_inverse_transformed_test_df[changed_columns],
        )

    @pytest.mark.parametrize(
        "transform, dataset_name, expected_changes",
        [
            # decomposition
            (
                ChangePointsSegmentationTransform(
                    in_column="target",
                    change_points_model=RupturesChangePointsModel(change_points_model=Binseg(), n_bkps=5),
                    out_column="res",
                ),
                "regular_ts",
                {},
            ),
            (
                ChangePointsTrendTransform(in_column="target"),
                "regular_ts",
                {},
            ),
            (
                ChangePointsTrendTransform(in_column="positive"),
                "ts_with_exog",
                {"change": {"positive"}},
            ),
            (
                ChangePointsLevelTransform(in_column="target"),
                "regular_ts",
                {},
            ),
            (
                ChangePointsLevelTransform(in_column="positive"),
                "ts_with_exog",
                {"change": {"positive"}},
            ),
            (LinearTrendTransform(in_column="target"), "regular_ts", {}),
            (LinearTrendTransform(in_column="positive"), "ts_with_exog", {"change": {"positive"}}),
            (TheilSenTrendTransform(in_column="target"), "regular_ts", {}),
            (TheilSenTrendTransform(in_column="positive"), "ts_with_exog", {"change": {"positive"}}),
            (STLTransform(in_column="target", period=7), "regular_ts", {}),
            (STLTransform(in_column="positive", period=7), "ts_with_exog", {"change": {"positive"}}),
            (DeseasonalityTransform(in_column="target", period=7), "regular_ts", {}),
            (DeseasonalityTransform(in_column="positive", period=7), "ts_with_exog", {"change": {"positive"}}),
            (
                TrendTransform(
                    in_column="target",
                    change_points_model=RupturesChangePointsModel(change_points_model=Binseg(), n_bkps=5),
                    out_column="res",
                ),
                "regular_ts",
                {},
            ),
            # encoders
            (LabelEncoderTransform(in_column="weekday", out_column="res"), "ts_with_exog", {}),
            (
                OneHotEncoderTransform(in_column="weekday", out_column="res"),
                "ts_with_exog",
                {},
            ),
            (MeanSegmentEncoderTransform(), "regular_ts", {}),
            (SegmentEncoderTransform(), "regular_ts", {}),
            # feature_selection
            (FilterFeaturesTransform(exclude=["year"]), "ts_with_exog", {}),
            (
                GaleShapleyFeatureSelectionTransform(relevance_table=StatisticsRelevanceTable(), top_k=2),
                "ts_with_exog",
                {},
            ),
            (
                MRMRFeatureSelectionTransform(
                    relevance_table=StatisticsRelevanceTable(), top_k=2, fast_redundancy=True
                ),
                "ts_with_exog",
                {},
            ),
            (
                MRMRFeatureSelectionTransform(
                    relevance_table=StatisticsRelevanceTable(), top_k=2, fast_redundancy=False
                ),
                "ts_with_exog",
                {},
            ),
            (
                TreeFeatureSelectionTransform(model=DecisionTreeRegressor(random_state=42), top_k=2),
                "ts_with_exog",
                {},
            ),
            # math
            (
                AddConstTransform(in_column="target", value=1, inplace=False, out_column="res"),
                "regular_ts",
                {},
            ),
            (AddConstTransform(in_column="target", value=1, inplace=True), "regular_ts", {}),
            (AddConstTransform(in_column="positive", value=1, inplace=True), "ts_with_exog", {"change": {"positive"}}),
            (
                LagTransform(in_column="target", lags=[1, 2, 3], out_column="res"),
                "regular_ts",
                {},
            ),
            (
                LambdaTransform(in_column="target", transform_func=lambda x: x + 1, inplace=False, out_column="res"),
                "regular_ts",
                {},
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
            (
                LambdaTransform(
                    in_column="positive",
                    transform_func=lambda x: x + 1,
                    inverse_transform_func=lambda x: x - 1,
                    inplace=True,
                ),
                "ts_with_exog",
                {"change": {"positive"}},
            ),
            (LogTransform(in_column="target", inplace=False, out_column="res"), "positive_ts", {}),
            (LogTransform(in_column="target", inplace=True), "positive_ts", {}),
            (LogTransform(in_column="positive", inplace=True), "ts_with_exog", {"change": {"positive"}}),
            (
                DifferencingTransform(in_column="target", inplace=False, out_column="res"),
                "regular_ts",
                {},
            ),
            (MADTransform(in_column="target", window=14, out_column="res"), "regular_ts", {}),
            (MaxTransform(in_column="target", window=14, out_column="res"), "regular_ts", {}),
            (MeanTransform(in_column="target", window=14, out_column="res"), "regular_ts", {}),
            (MedianTransform(in_column="target", window=14, out_column="res"), "regular_ts", {}),
            (
                MinMaxDifferenceTransform(in_column="target", window=14, out_column="res"),
                "regular_ts",
                {},
            ),
            (MinTransform(in_column="target", window=14, out_column="res"), "regular_ts", {}),
            (
                QuantileTransform(in_column="target", quantile=0.9, window=14, out_column="res"),
                "regular_ts",
                {},
            ),
            (StdTransform(in_column="target", window=14, out_column="res"), "regular_ts", {}),
            (SumTransform(in_column="target", window=14, out_column="res"), "regular_ts", {}),
            (
                BoxCoxTransform(in_column="target", mode="per-segment", inplace=False, out_column="res"),
                "positive_ts",
                {},
            ),
            (BoxCoxTransform(in_column="target", mode="per-segment", inplace=True), "positive_ts", {}),
            (
                BoxCoxTransform(in_column="positive", mode="per-segment", inplace=True),
                "ts_with_exog",
                {"change": {"positive"}},
            ),
            (
                BoxCoxTransform(in_column="target", mode="macro", inplace=False, out_column="res"),
                "positive_ts",
                {},
            ),
            (BoxCoxTransform(in_column="target", mode="macro", inplace=True), "positive_ts", {}),
            (
                BoxCoxTransform(in_column="positive", mode="macro", inplace=True),
                "ts_with_exog",
                {"change": {"positive"}},
            ),
            (
                MaxAbsScalerTransform(in_column="target", mode="per-segment", inplace=False, out_column="res"),
                "regular_ts",
                {},
            ),
            (MaxAbsScalerTransform(in_column="target", mode="per-segment", inplace=True), "regular_ts", {}),
            (
                MaxAbsScalerTransform(in_column="positive", mode="per-segment", inplace=True),
                "ts_with_exog",
                {"change": {"positive"}},
            ),
            (
                MaxAbsScalerTransform(in_column="target", mode="macro", inplace=False, out_column="res"),
                "regular_ts",
                {},
            ),
            (
                MaxAbsScalerTransform(in_column="target", mode="macro", inplace=True),
                "regular_ts",
                {},
            ),
            (
                MaxAbsScalerTransform(in_column="positive", mode="macro", inplace=True),
                "ts_with_exog",
                {"change": {"positive"}},
            ),
            (
                MinMaxScalerTransform(in_column="target", mode="per-segment", inplace=False, out_column="res"),
                "regular_ts",
                {},
            ),
            (MinMaxScalerTransform(in_column="target", mode="per-segment", inplace=True), "regular_ts", {}),
            (
                # setting clip=False is important
                MinMaxScalerTransform(in_column="positive", mode="per-segment", clip=False, inplace=True),
                "ts_with_exog",
                {"change": {"positive"}},
            ),
            (
                MinMaxScalerTransform(in_column="target", mode="macro", inplace=False, out_column="res"),
                "regular_ts",
                {},
            ),
            (
                MinMaxScalerTransform(in_column="target", mode="macro", inplace=True),
                "regular_ts",
                {},
            ),
            (
                # setting clip=False is important
                MinMaxScalerTransform(in_column="positive", mode="macro", clip=False, inplace=True),
                "ts_with_exog",
                {"change": {"positive"}},
            ),
            (
                RobustScalerTransform(in_column="target", mode="per-segment", inplace=False, out_column="res"),
                "regular_ts",
                {},
            ),
            (RobustScalerTransform(in_column="target", mode="per-segment", inplace=True), "regular_ts", {}),
            (
                RobustScalerTransform(in_column="positive", mode="per-segment", inplace=True),
                "ts_with_exog",
                {"change": {"positive"}},
            ),
            (
                RobustScalerTransform(in_column="target", mode="macro", inplace=False, out_column="res"),
                "regular_ts",
                {},
            ),
            (
                RobustScalerTransform(in_column="target", mode="macro", inplace=True),
                "regular_ts",
                {},
            ),
            (
                RobustScalerTransform(in_column="positive", mode="macro", inplace=True),
                "ts_with_exog",
                {"change": {"positive"}},
            ),
            (
                StandardScalerTransform(in_column="target", mode="per-segment", inplace=False, out_column="res"),
                "regular_ts",
                {},
            ),
            (StandardScalerTransform(in_column="target", mode="per-segment", inplace=True), "regular_ts", {}),
            (
                StandardScalerTransform(in_column="positive", mode="per-segment", inplace=True),
                "ts_with_exog",
                {"change": {"positive"}},
            ),
            (
                StandardScalerTransform(in_column="target", mode="macro", inplace=False, out_column="res"),
                "regular_ts",
                {},
            ),
            (
                StandardScalerTransform(in_column="target", mode="macro", inplace=True),
                "regular_ts",
                {},
            ),
            (
                StandardScalerTransform(in_column="positive", mode="macro", inplace=True),
                "ts_with_exog",
                {"change": {"positive"}},
            ),
            (
                YeoJohnsonTransform(in_column="target", mode="per-segment", inplace=False, out_column="res"),
                "regular_ts",
                {},
            ),
            (YeoJohnsonTransform(in_column="target", mode="per-segment", inplace=True), "regular_ts", {}),
            (
                YeoJohnsonTransform(in_column="positive", mode="per-segment", inplace=True),
                "ts_with_exog",
                {"change": {"positive"}},
            ),
            (
                YeoJohnsonTransform(in_column="target", mode="macro", inplace=False, out_column="res"),
                "regular_ts",
                {},
            ),
            (YeoJohnsonTransform(in_column="target", mode="macro", inplace=True), "regular_ts", {}),
            (
                YeoJohnsonTransform(in_column="positive", mode="macro", inplace=True),
                "ts_with_exog",
                {"change": {"positive"}},
            ),
            # missing_values
            (
                ResampleWithDistributionTransform(
                    in_column="regressor_exog", distribution_column="target", inplace=False, out_column="res"
                ),
                "ts_to_resample",
                {},
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
                {},
            ),
            (
                FourierTransform(period=7, order=2, out_column="res"),
                "regular_ts",
                {},
            ),
            (HolidayTransform(out_column="res", mode="binary"), "regular_ts", {}),
            (HolidayTransform(out_column="res", mode="category"), "regular_ts", {}),
            (
                TimeFlagsTransform(out_column="res"),
                "regular_ts",
                {},
            ),
            (SpecialDaysTransform(), "regular_ts", {}),
        ],
    )
    def test_inverse_transform_future_without_target(self, transform, dataset_name, expected_changes, request):
        ts = request.getfixturevalue(dataset_name)
        self._test_inverse_transform_future_without_target(ts, transform, expected_changes=expected_changes)

    @pytest.mark.parametrize(
        "transform, dataset_name, expected_changes",
        [
            (DifferencingTransform(in_column="positive", inplace=True), "ts_with_exog", {"change": {"positive"}}),
            (DifferencingTransform(in_column="target", inplace=True), "regular_ts", {}),
        ],
    )
    def test_inverse_transform_future_without_target_fail_difference(
        self, transform, dataset_name, expected_changes, request
    ):
        ts = request.getfixturevalue(dataset_name)
        with pytest.raises(ValueError, match="Test should go after the train without gaps"):
            self._test_inverse_transform_future_without_target(ts, transform, expected_changes=expected_changes)

    # It is the only transform that doesn't change values back during `inverse_transform`
    @to_be_fixed(AssertionError)
    @pytest.mark.parametrize(
        "transform, dataset_name, expected_changes",
        [
            (
                ResampleWithDistributionTransform(
                    in_column="regressor_exog", distribution_column="target", inplace=True
                ),
                "ts_to_resample",
                {"change": {"regressor_exog"}},
            ),
        ],
    )
    def test_inverse_transform_future_without_target_fail_resample(
        self, transform, dataset_name, expected_changes, request
    ):
        ts = request.getfixturevalue(dataset_name)
        self._test_inverse_transform_future_without_target(ts, transform, expected_changes=expected_changes)

    @to_be_fixed(raises=Exception)
    @pytest.mark.parametrize(
        "transform, dataset_name, expected_changes",
        [
            # feature_selection
            (FilterFeaturesTransform(exclude=["year"], return_features=True), "ts_with_exog", {"create": {"year"}}),
            (
                GaleShapleyFeatureSelectionTransform(
                    relevance_table=StatisticsRelevanceTable(), top_k=2, return_features=True
                ),
                "ts_with_exog",
                {"create": {"month", "year", "weekday"}},
            ),
            (
                MRMRFeatureSelectionTransform(
                    relevance_table=StatisticsRelevanceTable(), top_k=2, return_features=True, fast_redundancy=True
                ),
                "ts_with_exog",
                {"create": {"weekday", "monthday", "positive"}},
            ),
            (
                MRMRFeatureSelectionTransform(
                    relevance_table=StatisticsRelevanceTable(), top_k=2, return_features=True, fast_redundancy=False
                ),
                "ts_with_exog",
                {"create": {"weekday", "monthday", "positive"}},
            ),
            (
                TreeFeatureSelectionTransform(
                    model=DecisionTreeRegressor(random_state=42), top_k=2, return_features=True
                ),
                "ts_with_exog",
                {"create": {"year", "month", "weekday"}},
            ),
        ],
    )
    def test_inverse_transform_future_without_target_failed_error(
        self, transform, dataset_name, expected_changes, request
    ):
        ts = request.getfixturevalue(dataset_name)
        self._test_inverse_transform_future_without_target(ts, transform, expected_changes=expected_changes)
