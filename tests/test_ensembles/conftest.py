import pytest

from etna.ensembles import VotingEnsemble
from etna.models import CatBoostModelPerSegment
from etna.models import NaiveModel
from etna.models import ProphetModel
from etna.pipeline import Pipeline
from etna.transforms import LagTransform


@pytest.fixture
def catboost_pipeline() -> Pipeline:
    """Generate pipeline with CatBoostModelMultiSegment."""
    pipeline = Pipeline(
        model=CatBoostModelPerSegment(), transforms=[LagTransform(in_column="target", lags=[10, 11, 12])], horizon=7
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
def ensemble_pipeline(
    catboost_pipeline: Pipeline, prophet_pipeline: Pipeline, naive_pipeline_1: Pipeline
) -> VotingEnsemble:
    pipeline = VotingEnsemble(pipelines=[catboost_pipeline, prophet_pipeline, naive_pipeline_1])
    return pipeline
