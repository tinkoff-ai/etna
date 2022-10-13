from itertools import compress
from typing import List
from typing import Optional
from typing import Set

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.metrics import roc_auc_score

from etna.experimental.classification.base import PickleSerializable
from etna.experimental.classification.feature_extraction.base import BaseTimeSeriesFeatureExtractor
from etna.loggers import tslogger


class TimeSeriesBinaryClassifier(PickleSerializable):
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
        threshold:
            Positive class probability threshold.
        """
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.threshold = threshold
        self._classes: Optional[Set[int]] = None

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
        self._classes = set(map(int, y))
        x_tr = self.feature_extractor.fit_transform(x, y)
        self.classifier.fit(x_tr, y)
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
        y_prob_pred = self.predict_proba(x)
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
        if self._classes is None:
            raise ValueError("Classifier is not fitted!")

        x_tr = self.feature_extractor.transform(x)
        y_probs = self.classifier.predict_proba(x_tr)[:, 0]
        if 0 in self._classes:
            y_probs = 1 - y_probs
        return y_probs

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

            self.fit(x_train, y_train)
            y_pred = self.predict_proba(x_test)
            score = roc_auc_score(y_true=y_test, y_score=y_pred)
            cv_scores.append(score)

        mean_score = float(np.mean(cv_scores))
        tslogger.start_experiment(job_type="metrics", group="all")
        tslogger.log({"AUC": mean_score})
        tslogger.finish_experiment()

        return mean_score
