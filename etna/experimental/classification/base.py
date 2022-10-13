from itertools import compress
from typing import List

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.metrics import roc_auc_score

from etna.experimental.classification.feature_extraction.base import BaseTimeSeriesFeatureExtractor
from etna.loggers import tslogger


class TimeSeriesBinaryClassifier:
    """Class for holding time series binary classification."""

    def __init__(
        self, feature_extractor: BaseTimeSeriesFeatureExtractor, classifier: ClassifierMixin, threshold: float = 0.5
    ):
        """Init TimeSeriesClassifier with given parameters.

        Parameters
        ----------
        feature_extractor:
           Instance of time series feature extractor.
        classifier:
           Instance of classifier with sklearn interface.
        """
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.threshold = threshold

    def fit(self, x: List[np.ndarray], y: np.ndarray) -> "TimeSeriesBinaryClassifier":
        """Fit the classifier.

        Parameters
        ----------
        x:
            Array with time series.
        y:
            Array of class labels.

        Returns
        -------
        :
            Fitted instance of classifier.
        """
        x_tr = self.feature_extractor.fit_transform(x=x, y=y)
        self.classifier.fit(x=x_tr, y=y)
        return self

    def predict(self, x: List[np.ndarray]) -> np.ndarray:
        """Predict classes with threshold.

        Parameters
        ----------
        x:
           Array with time series.

        Returns
        -------
        :
            Array with predicted labels.
        """
        y_prob_pred = self.predict_proba(x=x)
        y_pred = (y_prob_pred > self.threshold).astype(int)
        return y_pred

    def predict_proba(self, x: List[np.ndarray]) -> np.ndarray:
        """Predict probabilities of the positive class.

        Parameters
        ----------
        x:
            Array with time series.

        Returns
        -------
        :
            Probabilities for classes.
        """
        x_tr = self.feature_extractor.transform(x=x)
        y_prob_pred = self.classifier.predict_proba(x=x_tr)
        return y_prob_pred

    def masked_crossval_score(self, x: List[np.ndarray], y: np.ndarray, mask: np.ndarray) -> float:
        """Calculate roc-auc on cross-validation.

        Parameters
        ----------
        x:
            Array with time series.
        y:
            Array of class labels.
        mask:
            Fold mask (array where for each element there is a label of its fold)

        Returns
        -------
        :
            Mean roc-auc score.
        """
        cv_scores = []
        for fold in np.unique(mask):
            x_train, y_train = list(compress(data=x, selectors=mask != fold)), y[mask != fold]
            x_test, y_test = list(compress(data=x, selectors=mask == fold)), y[mask == fold]

            self.fit(x=x_train, y=y_train)
            y_pred = self.predict_proba(x=x_test)
            score = roc_auc_score(y_true=y_test, y_score=y_pred)
            cv_scores.append(score)

        mean_score = float(np.mean(cv_scores))
        tslogger.start_experiment(job_type="metrics", group="all")
        tslogger.log({"AUC": mean_score})
        tslogger.finish_experiment()

        return mean_score
