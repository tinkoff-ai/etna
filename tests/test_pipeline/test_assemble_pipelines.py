import pytest

from etna.models import LinearPerSegmentModel
from etna.pipeline import Pipeline
from etna.pipeline import assemble_pipelines
from etna.transforms import TrendTransform


def test_not_equal_lengths():
    models = [LinearPerSegmentModel()]
    transforms = [TrendTransform(in_column="target")]
    horizons = [1, 2, 3]
    with pytest.raises(ValueError, match="len of sequences model and horizons must be equal"):
        _ = assemble_pipelines(models, transforms, horizons)


@pytest.mark.parametrize(
    "models", [[LinearPerSegmentModel(), LinearPerSegmentModel(), LinearPerSegmentModel()], LinearPerSegmentModel()]
)
def test_output_pipelines(models):
    transforms = [TrendTransform(in_column="target")]
    horizons = [1, 2, 3]
    pipelines = assemble_pipelines(models, transforms, horizons)
    assert len(pipelines) == len(horizons)
    for pipeline in pipelines:
        assert isinstance(pipeline, Pipeline)
