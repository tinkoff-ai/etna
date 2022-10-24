import numpy as np
import pytest
from sklearn.neighbors import KNeighborsClassifier
from tsfresh.feature_extraction.settings import MinimalFCParameters

from etna.experimental.classification.classification import TimeSeriesBinaryClassifier
from etna.experimental.classification.feature_extraction.tsfresh import TSFreshFeatureExtractor


def test_predict_proba_format(x_y):
    x, y = x_y
    clf = TimeSeriesBinaryClassifier(
        feature_extractor=TSFreshFeatureExtractor(default_fc_parameters=MinimalFCParameters()),
        classifier=KNeighborsClassifier(),
    )
    clf.fit(x, y)
    y_probs = clf.predict_proba(x)
    assert y_probs.shape == y.shape


def test_predict_format(x_y):
    x, y = x_y
    clf = TimeSeriesBinaryClassifier(
        feature_extractor=TSFreshFeatureExtractor(default_fc_parameters=MinimalFCParameters()),
        classifier=KNeighborsClassifier(),
    )
    clf.fit(x, y)
    y_pred = clf.predict(x)
    assert y_pred.shape == y.shape


@pytest.mark.parametrize("y", [(np.zeros(5)), (np.ones(5))])
def test_predict_single_class_on_fit(x_y, y):
    x, _ = x_y
    clf = TimeSeriesBinaryClassifier(
        feature_extractor=TSFreshFeatureExtractor(default_fc_parameters=MinimalFCParameters()),
        classifier=KNeighborsClassifier(),
    )
    clf.fit(x, y)
    y_pred = clf.predict(x)
    np.testing.assert_array_equal(y_pred, y)


def test_masked_crossval_score(many_time_series, folds=np.array([0, 0, 0, 1, 1, 1]), expected_score=1):
    """Test for masked_crossval_score method."""
    x, y = many_time_series
    x.extend(x)
    y = np.concatenate((y, y))
    clf = TimeSeriesBinaryClassifier(
        feature_extractor=TSFreshFeatureExtractor(default_fc_parameters=MinimalFCParameters()),
        classifier=KNeighborsClassifier(n_neighbors=1),
    )
    scores = clf.masked_crossval_score(x=x, y=y, mask=folds)
    for score in scores.values():
        assert np.mean(score) == expected_score


def test_dump_load_pipeline(x_y, tmp_path):
    x, y = x_y
    path = tmp_path / "tmp.pkl"
    clf = TimeSeriesBinaryClassifier(
        feature_extractor=TSFreshFeatureExtractor(default_fc_parameters=MinimalFCParameters()),
        classifier=KNeighborsClassifier(),
    )
    clf.fit(x, y)
    y_probs_original = clf.predict_proba(x)

    clf.dump(path=path)
    clf = TimeSeriesBinaryClassifier.load(path=path)
    y_probs_loaded = clf.predict_proba(x)

    np.testing.assert_array_equal(y_probs_original, y_probs_loaded)
