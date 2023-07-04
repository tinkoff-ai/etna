from etna import SETTINGS

if SETTINGS.classification_required:
    from etna.experimental.classification.feature_extraction.base import BaseTimeSeriesFeatureExtractor
    from etna.experimental.classification.feature_extraction.tsfresh import TSFreshFeatureExtractor
    from etna.experimental.classification.feature_extraction.weasel import WEASELFeatureExtractor
