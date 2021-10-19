from copy import deepcopy
from typing import List

import pandas as pd
import pytest
from joblib import Parallel
from joblib import delayed

from etna.ensembles import StackingEnsemble
from etna.ensembles import VotingEnsemble
from etna.models import CatBoostModelPerSegment
from etna.models import NaiveModel
from etna.models import ProphetModel
from etna.pipeline import Pipeline
from etna.transforms import DateFlagsTransform
from etna.transforms import LagTransform


@pytest.fixture
def catboost_pipeline() -> Pipeline:
    """Generate pipeline with CatBoostModelMultiSegment."""
    pipeline = Pipeline(
        model=CatBoostModelPerSegment(),
        transforms=[LagTransform(in_column="target", lags=[10, 11, 12])],
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
def stacking_ensemble_pipeline(
    catboost_pipeline: Pipeline, prophet_pipeline: Pipeline, naive_pipeline_1: Pipeline
) -> StackingEnsemble:
    pipeline = StackingEnsemble(pipelines=[catboost_pipeline, prophet_pipeline, naive_pipeline_1])
    return pipeline


@pytest.fixture
def naive_featured_pipeline_1() -> Pipeline:
    """Generate pipeline with NaiveModel(1)."""
    pipeline = Pipeline(model=NaiveModel(1), transforms=[LagTransform(lags=[10], in_column="target")], horizon=7)
    return pipeline


@pytest.fixture
def naive_featured_pipeline_2() -> Pipeline:
    """Generate pipeline with NaiveModel(2)."""
    pipeline = Pipeline(model=NaiveModel(2), transforms=[DateFlagsTransform()], horizon=7)
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
