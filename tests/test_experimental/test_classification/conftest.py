import numpy as np
import pytest


@pytest.fixture()
def x_y(random_seed):
    x = np.random.uniform(low=0, high=10, size=(5, 7))
    y = np.array([1, 0, 0, 1, 0])
    return x, y


@pytest.fixture()
def many_time_series():
    x = [np.array([1.0, 2.0, 3.0, 4.0]), np.array([4.0, 4.0, 4.0]), np.array([5.0, 5.0, 5.0, 6.0, 7.0, 8.0, 9.0])]
    y = np.array([1, 0, 1])
    return x, y
