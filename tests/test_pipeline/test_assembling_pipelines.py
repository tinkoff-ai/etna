import pytest

from etna.models import LinearPerSegmentModel
from etna.pipeline import Pipeline
from etna.pipeline import assemble_pipelines
from etna.transforms import LagTransform
from etna.transforms import TrendTransform


@pytest.mark.parametrize(
    "models, transforms, horizons, message",
    [
        (
            [LinearPerSegmentModel(), LinearPerSegmentModel()],
            [TrendTransform(in_column="target")],
            [1, 2, 3],
            "Lengths of the result models is not equals to horizons or transforms",
        ),
        (
            [LinearPerSegmentModel(), LinearPerSegmentModel(), LinearPerSegmentModel()],
            [
                TrendTransform(in_column="target"),
                [TrendTransform(in_column="target"), TrendTransform(in_column="target")],
            ],
            [1, 2, 3],
            "Lengths of the result transforms is not equals to models or horizons",
        ),
        (
            [LinearPerSegmentModel(), LinearPerSegmentModel(), LinearPerSegmentModel()],
            [TrendTransform(in_column="target")],
            [1, 2],
            "Lengths of the result horizons is not equals to models or transforms",
        ),
    ],
)
def test_not_equal_lengths(models, transforms, horizons, message):
    with pytest.raises(ValueError, match=message):
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
        (
            [LinearPerSegmentModel(), LinearPerSegmentModel()],
            [
                TrendTransform(in_column="target"),
                [
                    LagTransform(lags=[1, 2, 3], in_column="target"),
                    LagTransform(lags=[2, 3, 4], in_column="target"),
                ],
                [
                    LagTransform(lags=[1, 2, 3], in_column="target"),
                    LagTransform(lags=[2, 3, 4], in_column="target"),
                ],
            ],
            [1, 2],
            2,
        ),
        ([LinearPerSegmentModel(), LinearPerSegmentModel()], [TrendTransform(in_column="target")], [1, 2], 2),
    ],
)
def test_output_pipelines(models, transforms, horizons, expected_len):
    pipelines = assemble_pipelines(models, transforms, horizons)
    assert len(pipelines) == expected_len
    for pipeline in pipelines:
        assert isinstance(pipeline, Pipeline)


@pytest.mark.parametrize(
    "models, transforms, horizons, expected_transforms_lens",
    [
        (
            [LinearPerSegmentModel(), LinearPerSegmentModel()],
            [TrendTransform(in_column="target"), [LagTransform(lags=[1, 2, 3], in_column="target"), None]],
            [1, 2],
            [2, 1],
        ),
        (
            [LinearPerSegmentModel(), LinearPerSegmentModel()],
            [None, [None, LagTransform(lags=[1, 2, 3], in_column="target")]],
            [1, 2],
            [0, 1],
        ),
        (
            [LinearPerSegmentModel(), LinearPerSegmentModel()],
            [
                LagTransform(lags=[1], in_column="target"),
                [LagTransform(lags=[1], in_column="target"), None],
                [LagTransform(lags=[1, 2, 3], in_column="target"), None],
            ],
            [1, 2],
            [3, 1],
        ),
    ],
)
def test_none_in_tranforms(models, transforms, horizons, expected_transforms_lens):
    pipelines = assemble_pipelines(models, transforms, horizons)
    assert [len(pipeline.transforms) for pipeline in pipelines] == expected_transforms_lens


def test_different_objects():
    models = LinearPerSegmentModel()
    transforms = [TrendTransform(in_column="target")]
    horizons = [1, 2, 3, 4, 5]
    pipelines = assemble_pipelines(models, transforms, horizons)
    assert len({id(pipeline.model) for pipeline in pipelines}) == len(pipelines)
    assert len({id(pipeline.horizon) for pipeline in pipelines}) == len(pipelines)


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
            [LinearPerSegmentModel(), LinearPerSegmentModel()],
            [
                TrendTransform(in_column="target"),
                [
                    LagTransform(lags=[1, 2, 3], in_column="target"),
                    LagTransform(lags=[2, 3, 4], in_column="target"),
                ],
                [
                    LagTransform(lags=[1, 2, 3], in_column="target"),
                    LagTransform(lags=[2, 3, 4], in_column="target"),
                ],
            ],
            [1, 2],
            3,
        ),
    ],
)
def test_transofrms_lengths(models, transforms, horizons, expected_len):
    pipelines = assemble_pipelines(models, transforms, horizons)
    for pipeline in pipelines:
        assert len(pipeline.transforms) == expected_len
