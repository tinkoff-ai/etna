import functools
import inspect
from abc import ABC
from abc import abstractmethod
from copy import deepcopy
from typing import List
from typing import Union

import pandas as pd

from etna.core.mixins import BaseMixin
from etna.datasets.tsdataset import TSDataset
from etna.loggers import tslogger


# TODO: make PyCharm see signature of decorated method
def log_decorator(f):
    """Add logging for method of the model."""
    patch_dict = {"function": f.__name__, "line": inspect.getsourcelines(f)[1], "name": inspect.getmodule(f).__name__}

    @functools.wraps(f)
    def wrapper(self, *args, **kwargs):
        tslogger.log(f"Calling method {f.__name__} of {self.__class__.__name__}", **patch_dict)
        result = f(self, *args, **kwargs)
        return result

    return wrapper


class Model(ABC, BaseMixin):
    """Class for holding specific models - autoregression and simple regressions."""

    def __init__(self):
        self._models = None

    @abstractmethod
    def fit(self, ts: TSDataset) -> "Model":
        """Fit model.

        Parameters
        ----------
        ts:
            Dataframe with features
        Returns
        -------
        """
        pass

    @abstractmethod
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
        pass

    @staticmethod
    def _forecast_segment(model, segment: Union[str, List[str]], ts: TSDataset) -> pd.DataFrame:
        segment_features = ts[:, segment, :]
        segment_features = segment_features.droplevel("segment", axis=1)
        segment_features = segment_features.reset_index()
        dates = segment_features["timestamp"]
        dates.reset_index(drop=True, inplace=True)
        segment_predict = model.predict(df=segment_features)
        segment_predict = pd.DataFrame({"target": segment_predict})
        segment_predict["segment"] = segment
        segment_predict["timestamp"] = dates
        return segment_predict


class PerSegmentModel(Model):
    """Class for holding specific models for persegment prediction."""

    def __init__(self, base_model):
        super(PerSegmentModel, self).__init__()
        self._base_model = base_model
        self._segments = None

    @log_decorator
    def fit(self, ts: TSDataset) -> "PerSegmentModel":
        """Fit model."""
        self._segments = ts.segments
        self._build_models()

        for segment in self._segments:
            model = self._models[segment]
            segment_features = ts[:, segment, :]
            segment_features = segment_features.dropna()
            segment_features = segment_features.droplevel("segment", axis=1)
            segment_features = segment_features.reset_index()
            model.fit(df=segment_features)
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
        """df = dataset.to_pandas(flatten=True)
        df = df[df["target"].isna()]
        if any(df.drop(columns="target").isna().any()):
            raise ValueError("Dataset contains NaN values on the forecast side", df.columns[df.isna().any()].tolist())
        df.sort_values(by=["segment", "timestamp"], inplace=True)"""

        result_list = list()
        for segment in self._segments:
            model = self._models[segment]

            segment_predict = self._forecast_segment(model, segment, ts)
            result_list.append(segment_predict)

        # need real case to test
        result_df = pd.concat(result_list, ignore_index=True)
        result_df = result_df.set_index(["timestamp", "segment"])
        df = ts.to_pandas(flatten=True)
        df = df.set_index(["timestamp", "segment"])
        df = df.combine_first(result_df).reset_index()

        df = TSDataset.to_dataset(df)
        ts.df = df
        ts.inverse_transform()
        return ts

    def _build_models(self):
        """Create a dict with models for each segment (if required)."""
        self._models = {}
        for segment in self._segments:
            self._models[segment] = deepcopy(self._base_model)
