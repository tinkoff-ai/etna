from sklearn.linear_model import LogisticRegression
from tsfresh.feature_extraction.settings import MinimalFCParameters

from etna.experimental.classification.feature_extraction import TSFreshFeatureExtractor


def test_fit_transform_format(x_y):
    x, y = x_y
    feature_extractor = TSFreshFeatureExtractor(default_fc_parameters=MinimalFCParameters())
    x_tr = feature_extractor.fit_transform(x, y)
    assert x_tr.shape == (5, 10)


def test_sklearn_classifier_fit_on_extracted_features(x_y):
    x, y = x_y
    model = LogisticRegression()
    feature_extractor = TSFreshFeatureExtractor(default_fc_parameters=MinimalFCParameters())
    x_tr = feature_extractor.fit_transform(x, y)
    model.fit(x_tr, y)
