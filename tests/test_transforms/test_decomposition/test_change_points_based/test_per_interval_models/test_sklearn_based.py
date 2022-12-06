import numpy as np
from sklearn.preprocessing import MinMaxScaler

from etna.transforms.decomposition.change_points_based.per_interval_models.sklearn_based import (
    SklearnPreprocessingPerIntervalModel,
)


def test_preprocessing2per_interval_model_adapter_fit_transform():
    model = SklearnPreprocessingPerIntervalModel(preprocessing=MinMaxScaler())
    train_data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    model.fit(features=train_data, target=train_data)
    prediction = model.predict(features=train_data)
    assert np.all(0 <= x <= 1 for x in prediction)


def test_preprocessing2per_interval_model_adapter_fit_transform_hard():
    model = SklearnPreprocessingPerIntervalModel(preprocessing=MinMaxScaler())
    train_data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    test_data = np.array([10, 20, 30])
    model.fit(features=train_data, target=train_data)
    prediction = model.predict(features=test_data)
    np.testing.assert_array_equal(prediction, [1, 2, 3])
