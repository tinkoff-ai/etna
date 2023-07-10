import warnings
from typing import List
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin

from etna.models.base import BaseAdapter
from etna.models.base import NonPredictionIntervalContextIgnorantAbstractModel
from etna.models.mixins import MultiSegmentModelMixin
from etna.models.mixins import NonPredictionIntervalContextIgnorantModelMixin
from etna.models.mixins import PerSegmentModelMixin


class _SklearnAdapter(BaseAdapter):
    def __init__(self, regressor: RegressorMixin):
        self.model = regressor
        self.regressor_columns: Optional[List[str]] = None

    def _check_not_used_columns(self, df: pd.DataFrame):
        if self.regressor_columns is None:
            raise ValueError("Something went wrong, regressor_columns is None!")

        columns_not_used = [col for col in df.columns if col not in ["target", "timestamp"] + self.regressor_columns]
        if columns_not_used:
            warnings.warn(
                message=f"This model doesn't work with exogenous features unknown in future. "
                f"Columns {columns_not_used} won't be used."
            )

    def _select_regressors(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Select data with regressors.

        During fit there can't be regressors with NaNs, they are removed at higher level.
        Look at the issue: https://github.com/tinkoff-ai/etna/issues/557

        During prediction without validation NaNs in regressors can lead to exception from the underlying model,
        but it depends on the model, so it was decided to not validate this.

        This model requires data to be in numeric dtype.
        """
        if self.regressor_columns is None:
            raise ValueError("Something went wrong, regressor_columns is None!")

        if self.regressor_columns:
            try:
                result = df[self.regressor_columns].apply(pd.to_numeric)
            except ValueError as e:
                raise ValueError(f"Only convertible to numeric features are allowed! Error: {str(e)}")
        else:
            raise ValueError("There are not features for fitting the model!")

        return result

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
        self._check_not_used_columns(df)
        features = self._select_regressors(df)
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
        :
            Array with predictions
        """
        features = self._select_regressors(df)
        pred = self.model.predict(features)
        return pred

    def predict_components(self, df: pd.DataFrame) -> pd.DataFrame:
        """Estimate prediction components.

        Parameters
        ----------
        df:
            features dataframe

        Returns
        -------
        :
            dataframe with prediction components
        """
        raise NotImplementedError("Prediction decomposition isn't currently implemented!")

    def get_model(self) -> RegressorMixin:
        """Get internal sklearn model that is used inside etna class.

        Returns
        -------
        :
           Internal model
        """
        return self.model


class SklearnPerSegmentModel(
    PerSegmentModelMixin,
    NonPredictionIntervalContextIgnorantModelMixin,
    NonPredictionIntervalContextIgnorantAbstractModel,
):
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


class SklearnMultiSegmentModel(
    MultiSegmentModelMixin,
    NonPredictionIntervalContextIgnorantModelMixin,
    NonPredictionIntervalContextIgnorantAbstractModel,
):
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
