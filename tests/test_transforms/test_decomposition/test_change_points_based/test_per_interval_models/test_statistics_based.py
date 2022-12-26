import numpy as np
import pytest

from etna.transforms.decomposition.change_points_based.per_interval_models.statistics_based import MeanPerIntervalModel
from etna.transforms.decomposition.change_points_based.per_interval_models.statistics_based import (
    MedianPerIntervalModel,
)


@pytest.mark.parametrize(
    "model_class,train_data,expected",
    (
        (MeanPerIntervalModel, np.array([0, 1, 2, 3, 4]), 2),
        (MedianPerIntervalModel, np.array([0, 1, 3, 3, 4]), 3),
    ),
)
def test_statistics_models_fit(model_class, train_data: np.ndarray, expected: float):
    model = model_class()
    model.fit(features=train_data, target=train_data)
    assert model._statistics_value == expected


@pytest.mark.parametrize(
    "model_class,train_data,test_data,expected",
    (
        (
            MeanPerIntervalModel,
            np.array([0, 1, 2, 3, 4]),
            np.array([1, 1, 1]),
            np.array([2, 2, 2]),
        ),
        (
            MedianPerIntervalModel,
            np.array([0, 1, 3, 3, 4]),
            np.array([1, 1, 1]),
            np.array([3, 3, 3]),
        ),
    ),
)
def test_statistics_models_fit_predict(
    model_class, train_data: np.ndarray, test_data: np.ndarray, expected: np.ndarray
):
    model = model_class()
    model.fit(features=train_data, target=train_data)
    prediction = model.predict(features=test_data)
    np.testing.assert_array_equal(prediction, expected)
