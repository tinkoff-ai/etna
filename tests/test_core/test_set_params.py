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
    pipeline = Pipeline(model=CatBoostMultiSegmentModel(iterations=1000, depth=10), transforms=(), horizon=5)
    pipeline = pipeline.set_params(
        **{"model.learning_rate": 1e-3, "model.depth": 8, "transforms": [AddConstTransform("column", 1)]}
    )
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
                "value": 1,
            }
        ],
    }
    obtained_dict = pipeline.to_dict()
    assert obtained_dict == expected_dict


def test_base_mixin_set_params_doesnt_change_params_inplace_estimator():
    catboost_model = CatBoostMultiSegmentModel(iterations=1000, depth=10)
    catboost_model.set_params(**{"learning_rate": 1e-3, "depth": 8})
    expected_dict = {
        "_target_": "etna.models.catboost.CatBoostMultiSegmentModel",
        "iterations": 1000,
        "depth": 10,
        "logging_level": "Silent",
        "kwargs": {},
    }
    obtained_dict = catboost_model.to_dict()
    assert obtained_dict == expected_dict


def test_base_mixin_set_params_doesnt_change_params_inplace_pipeline():
    pipeline = Pipeline(model=CatBoostMultiSegmentModel(iterations=1000, depth=10), transforms=(), horizon=5)
    pipeline.set_params(
        **{"model.learning_rate": 1e-3, "model.depth": 8, "transforms": [AddConstTransform("column", 1)]}
    )
    expected_dict = {
        "_target_": "etna.pipeline.pipeline.Pipeline",
        "horizon": 5,
        "model": {
            "_target_": "etna.models.catboost.CatBoostMultiSegmentModel",
            "depth": 10,
            "iterations": 1000,
            "kwargs": {},
            "logging_level": "Silent",
        },
        "transforms": (),
    }
    obtained_dict = pipeline.to_dict()
    assert obtained_dict == expected_dict


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


# TODO: add tests on list
@pytest.mark.parametrize("nested_dict, flat_dict, expected_dict", [
    ({"learning_rate": 1e-3}, {}, {"learning_rate": 1e-3}),
    ({}, {"depth": 8}, {"depth": 8}),
    ({}, {"model.depth": 8}, {"model": {"depth": 8}}),
    ({"learning_rate": 1e-3}, {"depth": 8}, {"learning_rate": 1e-3, "depth": 8}),
    ({"model": {"learning_rate": 1e-3}}, {"model.depth": 8}, {"model": {"learning_rate": 1e-3, "depth": 8}}),
    ({"learning_rate": 1e-3},  {"learning_rate": 3e-4}, {"learning_rate": 3e-4}),
])
def test_update_nested_dict_with_flat_dict(nested_dict, flat_dict, expected_dict):
    BaseMixin._update_nested_dict_with_flat_dict(nested_dict, flat_dict)
    assert nested_dict == expected_dict
