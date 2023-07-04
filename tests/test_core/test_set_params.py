import pytest

from etna.core import BaseMixin
from etna.models import CatBoostMultiSegmentModel
from etna.pipeline import Pipeline
from etna.transforms import AddConstTransform


def test_base_mixin_set_params_changes_params_estimator():
    catboost_model = CatBoostMultiSegmentModel(iterations=1000, depth=10)
    catboost_model = catboost_model.set_params(**{"learning_rate": 1e-3, "depth": 8})
    expected_dict = {
        "_target_": "etna.models.catboost.CatBoostMultiSegmentModel",
        "iterations": 1000,
        "depth": 8,
        "learning_rate": 1e-3,
        "logging_level": "Silent",
        "kwargs": {},
    }
    obtained_dict = catboost_model.to_dict()
    assert obtained_dict == expected_dict


def test_base_mixin_set_params_changes_params_pipeline():
    pipeline = Pipeline(
        model=CatBoostMultiSegmentModel(iterations=1000, depth=10),
        transforms=[AddConstTransform(in_column="column", value=1)],
        horizon=5,
    )
    pipeline = pipeline.set_params(**{"model.learning_rate": 1e-3, "model.depth": 8, "transforms.0.value": 2})
    expected_dict = {
        "_target_": "etna.pipeline.pipeline.Pipeline",
        "horizon": 5,
        "model": {
            "_target_": "etna.models.catboost.CatBoostMultiSegmentModel",
            "depth": 8,
            "iterations": 1000,
            "kwargs": {},
            "learning_rate": 0.001,
            "logging_level": "Silent",
        },
        "transforms": [
            {
                "_target_": "etna.transforms.math.add_constant.AddConstTransform",
                "in_column": "column",
                "inplace": True,
                "value": 2,
            }
        ],
    }
    obtained_dict = pipeline.to_dict()
    assert obtained_dict == expected_dict


def test_base_mixin_set_params_doesnt_change_params_inplace_estimator():
    catboost_model = CatBoostMultiSegmentModel(iterations=1000, depth=10)
    initial_dict = catboost_model.to_dict()
    catboost_model.set_params(**{"learning_rate": 1e-3, "depth": 8})
    obtained_dict = catboost_model.to_dict()
    assert obtained_dict == initial_dict


def test_base_mixin_set_params_doesnt_change_params_inplace_pipeline():
    pipeline = Pipeline(
        model=CatBoostMultiSegmentModel(iterations=1000, depth=10),
        transforms=[AddConstTransform(in_column="column", value=1)],
        horizon=5,
    )
    initial_dict = pipeline.to_dict()
    pipeline.set_params(**{"model.learning_rate": 1e-3, "model.depth": 8, "transforms.0.value": 2})
    obtained_dict = pipeline.to_dict()
    assert obtained_dict == initial_dict


def test_base_mixin_set_params_with_nonexistent_attributes_estimator():
    catboost_model = CatBoostMultiSegmentModel(iterations=1000, depth=10)
    with pytest.raises(TypeError, match=".*got an unexpected keyword argument.*"):
        catboost_model.set_params(**{"incorrect_attribute": 1e-3})


def test_base_mixin_set_params_with_nonexistent_not_nested_attribute_pipeline():
    pipeline = Pipeline(model=CatBoostMultiSegmentModel(iterations=1000, depth=10), transforms=(), horizon=5)
    with pytest.raises(TypeError, match=".*got an unexpected keyword argument.*"):
        pipeline.set_params(
            **{
                "incorrect_estimator": "value",
            }
        )


def test_base_mixin_set_params_with_nonexistent_nested_attribute_pipeline():
    pipeline = Pipeline(model=CatBoostMultiSegmentModel(iterations=1000, depth=10), transforms=(), horizon=5)
    with pytest.raises(TypeError, match=".*got an unexpected keyword argument.*"):
        pipeline.set_params(
            **{
                "model.incorrect_attribute": "value",
            }
        )


@pytest.mark.parametrize(
    "nested_structure, keys, value, expected_result",
    [
        ({}, ["key"], 1, {"key": 1}),
        ({"key": 1}, ["key"], 2, {"key": 2}),
        ({}, ["key_1", "key_2"], 1, {"key_1": {"key_2": 1}}),
        ({"key_1": {"key_2": 1}}, ["key_1", "key_2"], 2, {"key_1": {"key_2": 2}}),
        ({"key_1": 1}, ["key_1"], {1, 2}, {"key_1": {1, 2}}),
        ({"key_1": {"key_2": 1}}, ["key_1"], 2, {"key_1": 2}),
        ([1], ["0"], 2, [2]),
        ([{"key": 1}], ["0", "key"], 2, [{"key": 2}]),
        ({"key": [1]}, ["key", "0"], 2, {"key": [2]}),
        ((1,), ["0"], 2, (2,)),
        (({"key": 1},), ["0", "key"], 2, ({"key": 2},)),
        ({"key": (1,)}, ["key", "0"], 2, {"key": (2,)}),
    ],
)
def test_update_nested_structure(nested_structure, keys, value, expected_result):
    result = BaseMixin._update_nested_structure(nested_structure, keys, value)
    assert result == expected_result


@pytest.mark.parametrize(
    "nested_structure, keys, value",
    [
        ([1], ["0", "key"], 2),
        ({1}, ["key"], 1),
    ],
)
def test_update_nested_structure_fail(nested_structure, keys, value):
    with pytest.raises(ValueError, match=f"Structure to update is .* with type .*"):
        _ = BaseMixin._update_nested_structure(nested_structure, keys, value)
