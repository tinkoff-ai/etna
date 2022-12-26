import numpy as np
import pytest

from etna.transforms.decomposition.change_points_based.per_interval_models import ConstantPerIntervalModel


@pytest.mark.parametrize("value", (1.5, -100))
def test_constant_model_fit(value: float):
    model = ConstantPerIntervalModel()
    assert model.value is None
    model.fit([], [], value=value)
    assert model.value == value


@pytest.mark.parametrize("value", (1.5, -100))
def test_constant_model_predict(value: float):
    model = ConstantPerIntervalModel()
    model.fit([], [], value=value)
    features = np.ones(shape=(5,))
    prediction = model.predict(features=features)
    np.testing.assert_array_equal(prediction, features * value)
