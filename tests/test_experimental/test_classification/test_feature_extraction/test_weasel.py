import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from etna.experimental.classification.feature_extraction import WEASELFeatureExtractor
from etna.experimental.classification.feature_extraction.weasel import CustomWEASEL


@pytest.fixture()
def many_time_series_big():
    x = [np.random.randint(0, 1000, size=100)[:i] for i in range(50, 80)]
    y = [np.random.randint(0, 2, size=1)[0] for _ in range(50, 80)]
    return x, y


@pytest.fixture
def many_time_series_windowed_3_1():
    x = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [4.0, 4.0, 4.0],
            [5.0, 5.0, 5.0],
            [5.0, 5.0, 6.0],
            [5.0, 6.0, 7.0],
            [6.0, 7.0, 8.0],
            [7.0, 8.0, 9.0],
        ]
    )
    y = np.array([1, 1, 0, 1, 1, 1, 1, 1])
    cum_sum = [0, 2, 3, 8]
    return x, y, cum_sum


@pytest.fixture
def many_time_series_windowed_3_2():
    x = np.array([[2.0, 3.0, 4.0], [4.0, 4.0, 4.0], [5.0, 5.0, 5.0], [5.0, 6.0, 7.0], [7.0, 8.0, 9.0]])
    y = np.array([1, 0, 1, 1, 1])
    cum_sum = [0, 1, 2, 5]
    return x, y, cum_sum


@pytest.mark.parametrize(
    "window_size, window_step, expected",
    [(3, 1, "many_time_series_windowed_3_1"), (3, 2, "many_time_series_windowed_3_2")],
)
def test_windowed_view(many_time_series, window_size, window_step, expected, request):
    x, y = many_time_series
    x_windowed_expected, y_windowed_expected, n_windows_per_sample_cum_expected = request.getfixturevalue(expected)
    x_windowed, y_windowed, n_windows_per_sample_cum = CustomWEASEL._windowed_view(
        x=x, y=y, window_size=window_size, window_step=window_step
    )
    np.testing.assert_array_equal(x_windowed, x_windowed_expected)
    np.testing.assert_array_equal(y_windowed, y_windowed_expected)
    np.testing.assert_array_equal(n_windows_per_sample_cum, n_windows_per_sample_cum_expected)


def test_preprocessor_and_classifier(many_time_series_big):
    x, y = many_time_series_big
    model = LogisticRegression()
    feature_extractor = WEASELFeatureExtractor(padding_value=0, window_sizes=[10, 15])
    x_tr = feature_extractor.fit_transform(x, y)
    model.fit(x_tr, y)
