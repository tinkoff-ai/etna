from typing import Tuple

import numpy as np
import pandas as pd
import pytest
from numpy.random import RandomState
from scipy.stats import norm

from etna.datasets import TSDataset
from etna.models import CatBoostPerSegmentModel
from etna.models import NaiveModel
from etna.models.base import NonPredictionIntervalContextIgnorantAbstractModel
from etna.models.base import NonPredictionIntervalContextRequiredAbstractModel
from etna.models.base import PredictionIntervalContextIgnorantAbstractModel
from etna.models.base import PredictionIntervalContextRequiredAbstractModel
from etna.models.mixins import NonPredictionIntervalContextIgnorantModelMixin
from etna.models.mixins import NonPredictionIntervalContextRequiredModelMixin
from etna.models.mixins import PredictionIntervalContextIgnorantModelMixin
from etna.models.mixins import PredictionIntervalContextRequiredModelMixin
from etna.pipeline import Pipeline
from etna.transforms import LagTransform

INTERVAL_WIDTH = 0.95


@pytest.fixture
def catboost_pipeline() -> Pipeline:
    """Generate pipeline with CatBoostPerSegmentModel."""
    pipeline = Pipeline(
        model=CatBoostPerSegmentModel(),
        transforms=[LagTransform(in_column="target", lags=[10, 11, 12], out_column="regressor_lag_feature")],
        horizon=7,
    )
    return pipeline


@pytest.fixture
def naive_pipeline() -> Pipeline:
    """Generate pipeline with NaiveModel."""
    pipeline = Pipeline(
        model=NaiveModel(lag=7),
        horizon=7,
    )
    return pipeline


@pytest.fixture
def catboost_pipeline_big() -> Pipeline:
    """Generate pipeline with CatBoostPerSegmentModel."""
    pipeline = Pipeline(
        model=CatBoostPerSegmentModel(),
        transforms=[LagTransform(in_column="target", lags=[25, 26, 27], out_column="regressor_lag_feature")],
        horizon=24,
    )
    return pipeline


@pytest.fixture
def weekly_period_ts(n_repeats: int = 15, horizon: int = 7) -> Tuple["TSDataset", "TSDataset"]:
    segment_1 = [7.0, 7.0, 3.0, 1.0]
    segment_2 = [40.0, 70.0, 20.0, 10.0]
    ts_range = list(pd.date_range("2020-01-03", freq="1D", periods=n_repeats * len(segment_1)))
    df = pd.DataFrame(
        {
            "timestamp": ts_range * 2,
            "target": segment_1 * n_repeats + segment_2 * n_repeats,
            "segment": ["segment_1"] * n_repeats * len(segment_1) + ["segment_2"] * n_repeats * len(segment_2),
        }
    )
    ts_start = sorted(set(df.timestamp))[-horizon]
    train, test = (
        df[lambda x: x.timestamp < ts_start],
        df[lambda x: x.timestamp >= ts_start],
    )
    train = TSDataset(TSDataset.to_dataset(train), "D")
    test = TSDataset(TSDataset.to_dataset(test), "D")

    return train, test


@pytest.fixture
def splited_piecewise_constant_ts(
    first_constant_len=40, constant_1_1=7.0, constant_1_2=2.0, constant_2_1=50.0, constant_2_2=10.0, horizon=5
) -> Tuple["TSDataset", "TSDataset"]:

    segment_1 = [constant_1_1] * first_constant_len + [constant_1_2] * horizon * 2
    segment_2 = [constant_2_1] * first_constant_len + [constant_2_2] * horizon * 2

    quantile = norm.ppf(q=(1 + INTERVAL_WIDTH) / 2)
    sigma_1 = np.std([0.0] * horizon * 2 + [constant_1_1 - constant_1_2] * horizon)
    sigma_2 = np.std([0.0] * horizon * 2 + [constant_2_1 - constant_2_2] * horizon)
    lower = [x - sigma_1 * quantile for x in segment_1] + [x - sigma_2 * quantile for x in segment_2]
    upper = [x + sigma_1 * quantile for x in segment_1] + [x + sigma_2 * quantile for x in segment_2]

    ts_range = list(pd.date_range("2020-01-03", freq="1D", periods=len(segment_1)))
    lower_p = (1 - INTERVAL_WIDTH) / 2
    upper_p = (1 + INTERVAL_WIDTH) / 2
    df = pd.DataFrame(
        {
            "timestamp": ts_range * 2,
            "target": segment_1 + segment_2,
            f"target_{lower_p:.4g}": lower,
            f"target_{upper_p:.4g}": upper,
            "segment": ["segment_1"] * len(segment_1) + ["segment_2"] * len(segment_2),
        }
    )
    ts_start = sorted(set(df.timestamp))[-horizon]
    train, test = (
        df[lambda x: x.timestamp < ts_start],
        df[lambda x: x.timestamp >= ts_start],
    )
    train = TSDataset(TSDataset.to_dataset(train.drop([f"target_{lower_p:.4g}", f"target_{upper_p:.4g}"], axis=1)), "D")
    test = TSDataset(TSDataset.to_dataset(test), "D")
    return train, test


@pytest.fixture
def constant_ts(size=40) -> TSDataset:
    segment_1 = [7] * size
    segment_2 = [50] * size
    ts_range = list(pd.date_range("2020-01-03", freq="1D", periods=size))
    df = pd.DataFrame(
        {
            "timestamp": ts_range * 2,
            "target": segment_1 + segment_2,
            "segment": ["segment_1"] * size + ["segment_2"] * size,
        }
    )
    ts = TSDataset(TSDataset.to_dataset(df), "D")
    return ts


@pytest.fixture
def constant_noisy_ts(size=40, use_noise=True) -> TSDataset:
    noise = RandomState(seed=42).normal(scale=3, size=size * 2)
    segment_1 = [7] * size
    segment_2 = [50] * size
    ts_range = list(pd.date_range("2020-01-03", freq="1D", periods=size))
    df = pd.DataFrame(
        {
            "timestamp": ts_range * 2,
            "target": segment_1 + segment_2,
            "segment": ["segment_1"] * size + ["segment_2"] * size,
        }
    )
    if use_noise:
        df.loc[:, "target"] += noise
    ts = TSDataset(TSDataset.to_dataset(df), "D")
    return ts


@pytest.fixture
def step_ts() -> Tuple[TSDataset, pd.DataFrame, pd.DataFrame]:
    """Create TSDataset for backtest with expected metrics_df and forecast_df.

    This dataset has a constant values at train, fold_1, fold_2, fold_3,
    but in the next fragment value is increased by `add_value`.
    """
    horizon = 5
    n_folds = 3
    train_size = 20
    start_value = 10.0
    add_value = 5.0
    segment = "segment_1"
    timestamp = pd.date_range(start="2020-01-01", periods=train_size + n_folds * horizon, freq="D")
    target = [start_value] * train_size
    for i in range(n_folds):
        target += [target[-1] + add_value] * horizon

    df = pd.DataFrame({"timestamp": timestamp, "target": target, "segment": segment})
    ts = TSDataset(TSDataset.to_dataset(df), freq="D")

    metrics_df = pd.DataFrame(
        {"segment": [segment, segment, segment], "MAE": [add_value, add_value, add_value], "fold_number": [0, 1, 2]}
    )

    timestamp_forecast = timestamp[train_size:]
    target_forecast = []
    fold_number_forecast = []
    for i in range(n_folds):
        target_forecast += [start_value + i * add_value] * horizon
        fold_number_forecast += [i] * horizon
    forecast_df = pd.DataFrame(
        {"fold_number": fold_number_forecast, "target": target_forecast},
        index=timestamp_forecast,
    )
    forecast_df.columns = pd.MultiIndex.from_product(
        [[segment], ["fold_number", "target"]], names=("segment", "feature")
    )
    return ts, metrics_df, forecast_df


def _get_simple_df() -> pd.DataFrame:
    timerange = pd.date_range(start="2020-01-01", periods=10).to_list()
    df = pd.DataFrame({"timestamp": timerange + timerange})
    df["segment"] = ["segment_0"] * 10 + ["segment_1"] * 10
    df["target"] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] + [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    return df


@pytest.fixture
def simple_ts() -> TSDataset:
    df = _get_simple_df()
    df = TSDataset.to_dataset(df)
    ts = TSDataset(df, freq="D")
    return ts


@pytest.fixture
def simple_ts_starting_with_nans_one_segment(simple_ts) -> TSDataset:
    df = _get_simple_df()
    df = TSDataset.to_dataset(df)
    df.iloc[:2, 0] = np.NaN
    ts = TSDataset(df, freq="D")
    return ts


@pytest.fixture
def simple_ts_starting_with_nans_all_segments(simple_ts) -> TSDataset:
    df = _get_simple_df()
    df = TSDataset.to_dataset(df)
    df.iloc[:2, 0] = np.NaN
    df.iloc[:3, 1] = np.NaN
    ts = TSDataset(df, freq="D")
    return ts


@pytest.fixture
def masked_ts() -> TSDataset:
    timerange = pd.date_range(start="2020-01-01", periods=11).to_list()
    df = pd.DataFrame({"timestamp": timerange + timerange})
    df["segment"] = ["segment_0"] * 11 + ["segment_1"] * 11
    df["target"] = [0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1] + [0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0]
    df = TSDataset.to_dataset(df)
    ts = TSDataset(df, freq="D")
    return ts


@pytest.fixture
def ts_process_fold_forecast() -> TSDataset:
    timerange = pd.date_range(start="2020-01-01", periods=11).to_list()
    df = pd.DataFrame({"timestamp": timerange + timerange})
    df["segment"] = ["segment_0"] * 11 + ["segment_1"] * 11
    df["target"] = [1, 2, 3, 4, 100, 6, 7, 100, 100, 100, 100] + [1, 2, 3, 4, 5, 6, 7, 8, 9, -6, 11]
    df = TSDataset.to_dataset(df)
    ts = TSDataset(df, freq="D")
    return ts


class DummyModelBase:
    def fit(self, ts: TSDataset):
        return self

    def get_model(self) -> "DummyModelBase":
        return self

    @property
    def context_size(self) -> int:
        return 0

    def _forecast(self, ts: TSDataset, **kwargs) -> TSDataset:
        ts.loc[pd.IndexSlice[:], pd.IndexSlice[:, "target"]] = 100
        return ts

    def _predict(self, ts: TSDataset, **kwargs) -> TSDataset:
        ts.loc[pd.IndexSlice[:], pd.IndexSlice[:, "target"]] = 200
        return ts

    def _forecast_components(self, ts: TSDataset, **kwargs) -> pd.DataFrame:
        df = ts.to_pandas(flatten=True, features=["target"])
        df["target_component_a"] = 10
        df["target_component_b"] = 90
        df = df.drop(columns=["target"])
        df = TSDataset.to_dataset(df)
        return df

    def _predict_components(self, ts: TSDataset, **kwargs) -> pd.DataFrame:
        df = ts.to_pandas(flatten=True, features=["target"])
        df["target_component_a"] = 20
        df["target_component_b"] = 180
        df = df.drop(columns=["target"])
        df = TSDataset.to_dataset(df)
        return df


class NonPredictionIntervalContextIgnorantDummyModel(
    DummyModelBase, NonPredictionIntervalContextIgnorantModelMixin, NonPredictionIntervalContextIgnorantAbstractModel
):
    pass


class NonPredictionIntervalContextRequiredDummyModel(
    DummyModelBase, NonPredictionIntervalContextRequiredModelMixin, NonPredictionIntervalContextRequiredAbstractModel
):
    pass


class PredictionIntervalContextIgnorantDummyModel(
    DummyModelBase, PredictionIntervalContextIgnorantModelMixin, PredictionIntervalContextIgnorantAbstractModel
):
    pass


class PredictionIntervalContextRequiredDummyModel(
    DummyModelBase, PredictionIntervalContextRequiredModelMixin, PredictionIntervalContextRequiredAbstractModel
):
    pass


@pytest.fixture
def non_prediction_interval_context_ignorant_dummy_model():
    model = NonPredictionIntervalContextIgnorantDummyModel()
    return model


@pytest.fixture
def non_prediction_interval_context_required_dummy_model():
    model = NonPredictionIntervalContextRequiredDummyModel()
    return model


@pytest.fixture
def prediction_interval_context_ignorant_dummy_model():
    model = PredictionIntervalContextIgnorantDummyModel()
    return model


@pytest.fixture
def prediction_interval_context_required_dummy_model():
    model = PredictionIntervalContextRequiredDummyModel()
    return model
