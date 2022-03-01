import pytest

from etna.ensembles.stacking_ensemble import StackingEnsemble
from etna.pipeline import Pipeline

HORIZON = 7


def test_invalid_pipelines_number(catboost_pipeline: Pipeline):
    """Test StackingEnsemble behavior in case of invalid pipelines number."""
    with pytest.raises(ValueError, match="At least two pipelines are expected."):
        _ = StackingEnsemble(pipelines=[catboost_pipeline])


def test_get_horizon_pass(catboost_pipeline: Pipeline, prophet_pipeline: Pipeline):
    """Check that StackingEnsemble._get horizon works correctly in case of valid pipelines list."""
    horizon = StackingEnsemble._get_horizon(pipelines=[catboost_pipeline, prophet_pipeline])
    assert horizon == HORIZON


def test_get_horizon_fail(catboost_pipeline: Pipeline, naive_pipeline: Pipeline):
    """Check that StackingEnsemble._get horizon works correctly in case of invalid pipelines list."""
    with pytest.raises(ValueError, match="All the pipelines should have the same horizon."):
        _ = StackingEnsemble._get_horizon(pipelines=[catboost_pipeline, naive_pipeline])
