from typing import List
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin

from etna.datasets.tsdataset import TSDataset
from etna.models.base import Model
from etna.models.base import PerSegmentModel
from etna.models.base import log_decorator


class _SklearnModel:
    def __init__(self, regressor: RegressorMixin):
        self.model = regressor
        self.regressor_columns: Optional[List[str]] = None

    def fit(self, df: pd.DataFrame, regressors: List[str]) -> "_SklearnModel":
        self.regressor_columns = regressors
        try:
            features = df[self.regressor_columns].apply(pd.to_numeric)
        except ValueError:
            raise ValueError("Only convertible to numeric features are accepted!")
        target = df["target"]
        self.model.fit(features, target)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
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
        super().__init__(base_model=_SklearnModel(regressor=regressor))

    @log_decorator
    def fit(self, ts: TSDataset) -> "SklearnPerSegmentModel":
        """Fit model."""
        self._segments = ts.segments
        self._build_models()

        for segment in self._segments:
            model = self._models[segment]  # type: ignore
            segment_features = ts[:, segment, :]
            segment_features = segment_features.dropna()
            segment_features = segment_features.droplevel("segment", axis=1)
            segment_features = segment_features.reset_index()
            model.fit(df=segment_features, regressors=ts.regressors)
        return self


class SklearnMultiSegmentModel(Model):
    """Class for holding Sklearn model for all segments."""

    def __init__(self, regressor: RegressorMixin):
        """
        Create instance of SklearnMultiSegmentModel with given parameters.

        Parameters
        ----------
        regressor:
            sklearn model for regression
        """
        super().__init__()
        self._base_model = _SklearnModel(regressor=regressor)

    @log_decorator
    def fit(self, ts: TSDataset) -> "SklearnMultiSegmentModel":
        """Fit model."""
        df = ts.to_pandas(flatten=True)
        df = df.dropna()
        df = df.drop(columns="segment")
        self._base_model.fit(df=df, regressors=ts.regressors)
        return self

    @log_decorator
    def forecast(self, ts: TSDataset) -> TSDataset:
        """Make predictions.

        Parameters
        ----------
        ts:
            Dataframe with features
        Returns
        -------
        DataFrame
            Models result
        """
        horizon = len(ts.df)
        x = ts.to_pandas(flatten=True).drop(["segment"], axis=1)
        y = self._base_model.predict(x).reshape(-1, horizon).T
        ts.loc[:, pd.IndexSlice[:, "target"]] = y
        ts.inverse_transform()
        return ts
