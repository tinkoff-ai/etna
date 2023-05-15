from etna import SETTINGS

if SETTINGS.classification_required:
    from etna.experimental.classification.classification import TimeSeriesBinaryClassifier
    from etna.experimental.classification.predictability import PredictabilityAnalyzer
