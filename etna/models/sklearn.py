from typing import List
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin

from etna.models.base import MultiSegmentModel
from etna.models.base import PerSegmentModel


class _SklearnAdapter:
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
        self:
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

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Compute predictions from a Sklearn model.

        Parameters
        ----------
        df:
            Features dataframe

        Returns
        -------
        y_pred:
            Array with predictions
        """
        try:
            features = df[self.regressor_columns].apply(pd.to_numeric)
        except ValueError:
            raise ValueError("Only convertible to numeric features are accepted!")
        pred = self.model.predict(features)
        return pred


class SklearnPerSegmentModel(PerSegmentModel):
    """Class for holding per segment Sklearn model."""

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

    def __init__(self, regressor: RegressorMixin):
        """
        Create instance of SklearnMultiSegmentModel with given parameters.

        Parameters
        ----------
        regressor:
            Sklearn model for regression
        """
        super().__init__(base_model=_SklearnAdapter(regressor=regressor))
