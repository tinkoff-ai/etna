from copy import deepcopy
from typing import List
from typing import Tuple

import pandas as pd
import pytest
from joblib import Parallel
from joblib import delayed
from sklearn.tree import DecisionTreeRegressor

from etna.datasets import TSDataset
from etna.datasets import generate_ar_df
from etna.ensembles import StackingEnsemble
from etna.ensembles import VotingEnsemble
from etna.models import CatBoostPerSegmentModel
from etna.models import NaiveModel
from etna.models import ProphetModel
from etna.pipeline import Pipeline
from etna.transforms import DateFlagsTransform
from etna.transforms import LagTransform


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
def prophet_pipeline() -> Pipeline:
    """Generate pipeline with ProphetModel."""
    pipeline = Pipeline(model=ProphetModel(), transforms=[], horizon=7)
    return pipeline


@pytest.fixture
def naive_pipeline() -> Pipeline:
    """Generate pipeline with NaiveModel."""
    pipeline = Pipeline(model=NaiveModel(20), transforms=[], horizon=14)
    return pipeline


@pytest.fixture
def naive_pipeline_1() -> Pipeline:
    """Generate pipeline with NaiveModel(1)."""
    pipeline = Pipeline(model=NaiveModel(1), transforms=[], horizon=7)
    return pipeline


@pytest.fixture
def naive_pipeline_2() -> Pipeline:
    """Generate pipeline with NaiveModel(2)."""
    pipeline = Pipeline(model=NaiveModel(2), transforms=[], horizon=7)
    return pipeline


@pytest.fixture
def voting_ensemble_pipeline(
    catboost_pipeline: Pipeline, prophet_pipeline: Pipeline, naive_pipeline_1: Pipeline
) -> VotingEnsemble:
    pipeline = VotingEnsemble(pipelines=[catboost_pipeline, prophet_pipeline, naive_pipeline_1])
    return pipeline


@pytest.fixture
def voting_ensemble_naive(naive_pipeline_1: Pipeline, naive_pipeline_2: Pipeline) -> VotingEnsemble:
    pipeline = VotingEnsemble(pipelines=[naive_pipeline_1, naive_pipeline_2])
    return pipeline


@pytest.fixture
def stacking_ensemble_pipeline(
    catboost_pipeline: Pipeline, prophet_pipeline: Pipeline, naive_pipeline_1: Pipeline
) -> StackingEnsemble:
    pipeline = StackingEnsemble(pipelines=[catboost_pipeline, prophet_pipeline, naive_pipeline_1])
    return pipeline


@pytest.fixture
def naive_featured_pipeline_1() -> Pipeline:
    """Generate pipeline with NaiveModel(1)."""
    pipeline = Pipeline(
        model=NaiveModel(1),
        transforms=[LagTransform(lags=[10], in_column="target", out_column="regressor_lag_feature")],
        horizon=7,
    )
    return pipeline


@pytest.fixture
def naive_featured_pipeline_2() -> Pipeline:
    """Generate pipeline with NaiveModel(2)."""
    pipeline = Pipeline(
        model=NaiveModel(2), transforms=[DateFlagsTransform(out_column="regressor_dateflag")], horizon=7
    )
    return pipeline


@pytest.fixture
def forecasts_ts(
    example_tsds: "TSDataset", naive_featured_pipeline_1: Pipeline, naive_featured_pipeline_2: Pipeline
) -> List["TSDataset"]:
    ensemble = StackingEnsemble(pipelines=[naive_featured_pipeline_1, naive_featured_pipeline_2], features_to_use="all")
    forecasts = Parallel(n_jobs=ensemble.n_jobs, backend="multiprocessing", verbose=11)(
        delayed(ensemble._backtest_pipeline)(pipeline=pipeline, ts=deepcopy(example_tsds))
        for pipeline in ensemble.pipelines
    )
    return forecasts


@pytest.fixture
def targets(example_tsds: "TSDataset", forecasts_ts: List["TSDataset"]) -> pd.Series:
    y = pd.concat(
        [
            example_tsds[forecasts_ts[0].index.min() : forecasts_ts[0].index.max(), segment, "target"]
            for segment in example_tsds.segments
        ],
        axis=0,
    )
    return y


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
def naive_ensemble(horizon: int = 7) -> StackingEnsemble:
    naive_featured_pipeline_1 = Pipeline(
        model=NaiveModel(1),
        transforms=[LagTransform(lags=[horizon], in_column="target", out_column="regressor_lag_feature")],
        horizon=horizon,
    )
    naive_featured_pipeline_2 = Pipeline(model=NaiveModel(2), transforms=[DateFlagsTransform()], horizon=horizon)
    ensemble = StackingEnsemble(
        final_model=DecisionTreeRegressor(random_state=0),
        pipelines=[naive_featured_pipeline_1, naive_featured_pipeline_2],
        features_to_use="all",
    )
    return ensemble


@pytest.fixture
def ts_with_segment_named_target() -> TSDataset:
    df = generate_ar_df(periods=100, start_time="2020-01-01", n_segments=5, freq="D")
    df.loc[df["segment"] == "segment_0", "segment"] = "target"
    df_wide = TSDataset.to_dataset(df)
    ts = TSDataset(df=df_wide, freq="D")
    return ts
