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

    def fit(self, df: pd.DataFrame) -> "_SklearnModel":
        features = df.drop(columns=["timestamp", "target"])
        target = df["target"]
        self.model.fit(features, target)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        features = df.drop(columns=["timestamp", "target"])
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
        self._base_model.fit(df=df)
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
        result_list = list()
        for segment in ts.segments:
            segment_predict = self._forecast_segment(self._base_model, segment, ts)
            result_list.append(segment_predict)

        result_df = pd.concat(result_list, ignore_index=True)
        result_df = result_df.set_index(["timestamp", "segment"])

        df = ts.to_pandas(flatten=True)
        df = df.set_index(["timestamp", "segment"])
        df = df.combine_first(result_df).reset_index()

        df = TSDataset.to_dataset(df)
        ts.df = df
        ts.inverse_transform()
        return ts
