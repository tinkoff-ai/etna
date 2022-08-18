from typing import List
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin

from etna.models.base import BaseAdapter
from etna.models.base import MultiSegmentModel
from etna.models.base import PerSegmentModel
from etna.models.utils import select_prediction_size_timestamps


class _SklearnAdapter(BaseAdapter):
    def __init__(self, regressor: RegressorMixin):
        self.model = regressor
        self.regressor_columns: Optional[List[str]] = None

    def fit(self, df: pd.DataFrame, regressors: List[str]) -> "_SklearnAdapter":
        """
        Fit Sklearn model.

        Parameters
        ----------
        df:
            Features dataframe
        regressors:
            List of the columns with regressors

        Returns
        -------
        :
            Fitted model
        """
        self.regressor_columns = regressors
        try:
            features = df[self.regressor_columns].apply(pd.to_numeric)
        except ValueError:
            raise ValueError("Only convertible to numeric features are accepted!")
        target = df["target"]
        self.model.fit(features, target)
        return self

    def predict(self, df: pd.DataFrame, prediction_size: int) -> np.ndarray:
        """
        Compute predictions from a Sklearn model.

        Parameters
        ----------
        df:
            Features dataframe
        prediction_size:
            Number of last timestamps to leave after making prediction.
            Previous timestamps will be used as a context for models that require it.

        Returns
        -------
        :
            Array with predictions
        """
        try:
            features = df[self.regressor_columns].apply(pd.to_numeric)
        except ValueError:
            raise ValueError("Only convertible to numeric features are accepted!")
        pred = self.model.predict(features)
        pred = select_prediction_size_timestamps(
            prediction=pred, timestamp=df["timestamp"], prediction_size=prediction_size
        )
        return pred

    def get_model(self) -> RegressorMixin:
        """Get internal sklearn model that is used inside etna class.

        Returns
        -------
        :
           Internal model
        """
        return self.model


class SklearnPerSegmentModel(PerSegmentModel):
    """Class for holding per segment Sklearn model."""

    context_size = 0

    def __init__(self, regressor: RegressorMixin):
        """
        Create instance of SklearnPerSegmentModel with given parameters.

        Parameters
        ----------
        regressor:
            sklearn model for regression
        """
        super().__init__(base_model=_SklearnAdapter(regressor=regressor))


class SklearnMultiSegmentModel(MultiSegmentModel):
    """Class for holding Sklearn model for all segments."""

    context_size = 0

    def __init__(self, regressor: RegressorMixin):
        """
        Create instance of SklearnMultiSegmentModel with given parameters.

        Parameters
        ----------
        regressor:
            Sklearn model for regression
        """
        super().__init__(base_model=_SklearnAdapter(regressor=regressor))
