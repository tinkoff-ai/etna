from itertools import compress
from typing import Dict
from typing import List
from typing import Optional
from typing import Set

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score

from etna.core import BaseMixin
from etna.experimental.classification.base import PickleSerializable
from etna.experimental.classification.feature_extraction.base import BaseTimeSeriesFeatureExtractor
from etna.loggers import tslogger


class TimeSeriesBinaryClassifier(BaseMixin, PickleSerializable):
    """Class for holding time series binary classification."""

    NEGATIVE_CLASS = 0
    POSITIVE_CLASS = 1

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
        self._classes = set(y)
        if len(self._classes - {0, 1}) != 0:
            raise ValueError("Only the 0 - negative and 1 - positive are possible values for the class labels!")

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
        y_probs = self.classifier.predict_proba(x_tr)
        if self.NEGATIVE_CLASS in self._classes and self.POSITIVE_CLASS in self._classes:
            return y_probs[:, 1]
        elif self.NEGATIVE_CLASS in self._classes:
            return 1 - y_probs[:, 0]
        return y_probs[:, 0]

    def masked_crossval_score(self, x: List[np.ndarray], y: np.ndarray, mask: np.ndarray) -> Dict[str, list]:
        """Calculate classification metrics on cross-validation.

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
            Classification metrics for each fold
        """
        roc_auc_scores = []
        other_metrics = []
        for fold in np.unique(mask):
            x_train, y_train = list(compress(data=x, selectors=mask != fold)), y[mask != fold]
            x_test, y_test = list(compress(data=x, selectors=mask == fold)), y[mask == fold]

            self.fit(x_train, y_train)
            y_prob_pred = self.predict_proba(x_test)
            y_pred = (y_prob_pred > self.threshold).astype(int)
            roc_auc_scores.append(roc_auc_score(y_true=y_test, y_score=y_prob_pred))
            other_metrics.append(precision_recall_fscore_support(y_true=y_test, y_pred=y_pred, average="macro")[:-1])

        per_fold_metrics: Dict[str, list] = {metric: [] for metric in ["precision", "recall", "fscore"]}
        for fold_metrics in other_metrics:
            for i, metric in enumerate(["precision", "recall", "fscore"]):
                per_fold_metrics[metric].append(fold_metrics[i])
        per_fold_metrics["AUC"] = roc_auc_scores
        mean_metrics = {metric: float(np.mean(values)) for metric, values in per_fold_metrics.items()}

        tslogger.start_experiment(job_type="metrics", group="all")
        tslogger.log(mean_metrics)
        tslogger.finish_experiment()

        return per_fold_metrics
