import pytest

from etna.models import LinearPerSegmentModel
from etna.pipeline import Pipeline
from etna.pipeline import assemble_pipelines
from etna.transforms import LagTransform
from etna.transforms import TrendTransform


def test_not_equal_lengths():
    models = [LinearPerSegmentModel()]
    transforms = [TrendTransform(in_column="target")]
    horizons = [1, 2, 3]
    with pytest.raises(ValueError):
        _ = assemble_pipelines(models, transforms, horizons)


@pytest.mark.parametrize(
    "models, transforms, horizons, expected_len",
    [
        (LinearPerSegmentModel(), [TrendTransform(in_column="target")], 1, 1),
        (
            LinearPerSegmentModel(),
            [
                TrendTransform(in_column="target"),
                [LagTransform(lags=[1, 2, 3], in_column="target"), LagTransform(lags=[2, 3, 4], in_column="target")],
            ],
            [1, 2],
            2,
        ),
        (
            [LinearPerSegmentModel(), LinearPerSegmentModel()],
            [
                TrendTransform(in_column="target"),
                [LagTransform(lags=[1, 2, 3], in_column="target"), LagTransform(lags=[2, 3, 4], in_column="target")],
            ],
            1,
            2,
        ),
        (
            [LinearPerSegmentModel(), LinearPerSegmentModel(), LinearPerSegmentModel()],
            [
                TrendTransform(in_column="target"),
                [
                    LagTransform(lags=[1, 2, 3], in_column="target"),
                    LagTransform(lags=[2, 3, 4], in_column="target"),
                    None,
                ],
            ],
            [1, 2, 3],
            3,
        ),
    ],
)
def test_output_pipelines(models, transforms, horizons, expected_len):
    pipelines = assemble_pipelines(models, transforms, horizons)
    assert len(pipelines) == expected_len
    for pipeline in pipelines:
        assert isinstance(pipeline, Pipeline)


def test_none_in_tranforms():
    models = [LinearPerSegmentModel(), LinearPerSegmentModel()]
    transforms = [TrendTransform(in_column="target"), [LagTransform(lags=[1, 2, 3], in_column="target"), None]]
    horizons = [1, 2]
    expected_transforms_lens = {1, 2}
    pipelines = assemble_pipelines(models, transforms, horizons)
    assert {len(pipeline.transforms) for pipeline in pipelines} == expected_transforms_lens
