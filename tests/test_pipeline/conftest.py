import pytest

from etna.models import CatBoostModelPerSegment
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
def catboost_pipeline_big() -> Pipeline:
    """Generate pipeline with CatBoostModelMultiSegment."""
    pipeline = Pipeline(
        model=CatBoostModelPerSegment(), transforms=[LagTransform(in_column="target", lags=[25, 26, 27])], horizon=24
    )
    return pipeline
