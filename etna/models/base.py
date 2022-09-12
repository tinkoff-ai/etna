import functools
import inspect
from abc import ABC
from abc import abstractmethod
from copy import deepcopy
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import Optional
from typing import Sequence
from typing import Sized
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd

from etna import SETTINGS
from etna.core.mixins import BaseMixin
from etna.datasets.tsdataset import TSDataset
from etna.loggers import tslogger

if SETTINGS.torch_required:
    import torch
    from pytorch_lightning import LightningModule
    from pytorch_lightning import Trainer
    from torch.utils.data import DataLoader
    from torch.utils.data import Dataset
    from torch.utils.data import random_split
else:
    from unittest.mock import Mock

    LightningModule = Mock  # type: ignore


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


class AbstractModel(ABC, BaseMixin):
    """Interface for model with fit method."""

    @property
    @abstractmethod
    def context_size(self) -> int:
        """Context size of the model. Determines how many history points do we ask to pass to the model."""
        pass

    @abstractmethod
    def fit(self, ts: TSDataset) -> "AbstractModel":
        """Fit model.

        Parameters
        ----------
        ts:
            Dataset with features

        Returns
        -------
        :
            Model after fit
        """
        pass

    @abstractmethod
    def get_model(self) -> Union[Any, Dict[str, Any]]:
        """Get internal model/models that are used inside etna class.

        Internal model is a model that is used inside etna to forecast segments,
        e.g. :py:class:`catboost.CatBoostRegressor` or :py:class:`sklearn.linear_model.Ridge`.

        Returns
        -------
        :
            The result can be of two types:

            * if model is multi-segment, then the result is internal model

            * if model is per-segment, then the result is dictionary where key is segment and value is internal model

        """
        pass


class NonPredictionIntervalContextIgnorantAbstractModel(AbstractModel):
    """Interface for models that don't support prediction intervals and don't need context for prediction."""

    @property
    def context_size(self) -> int:
        """Context size of the model. Determines how many history points do we ask to pass to the model.

        Zero for this model.
        """
        return 0

    @abstractmethod
    def forecast(self, ts: TSDataset) -> TSDataset:
        """Make predictions.

        * If it is regression model, the results of ``forecast`` and ``predict`` are the same.

        * If it is autoregression model, this method will make autoregressive predictions.

        To understand how a particular model behaves look at its documentation.

        Parameters
        ----------
        ts:
            Dataset with features

        Returns
        -------
        :
            Dataset with predictions
        """
        pass

    @abstractmethod
    def predict(self, ts: TSDataset) -> TSDataset:
        """Make predictions.

        * If it is regression model, the results of ``forecast`` and ``predict`` are the same.

        * If it is autoregression model, this method will make predictions using true values
          instead of predicted on a previous step.
          It can be useful for making in-sample forecasts.

        To understand how a particular model behaves look at its documentation.

        Parameters
        ----------
        ts:
            Dataset with features

        Returns
        -------
        :
            Dataset with predictions
        """
        pass


class NonPredictionIntervalContextRequiredAbstractModel(AbstractModel):
    """Interface for models that don't support prediction intervals and need context for prediction."""

    @abstractmethod
    def forecast(self, ts: TSDataset, prediction_size: int) -> TSDataset:
        """Make predictions.

        * If it is regression model, the results of ``forecast`` and ``predict`` are the same.

        * If it is autoregression model, this method will make autoregressive predictions.

        To understand how a particular model behaves look at its documentation.

        Parameters
        ----------
        ts:
            Dataset with features
        prediction_size:
            Number of last timestamps to leave after making prediction.
            Previous timestamps will be used as a context for models that require it.

        Returns
        -------
        :
            Dataset with predictions
        """
        pass

    @abstractmethod
    def predict(self, ts: TSDataset, prediction_size: int) -> TSDataset:
        """Make predictions.

        * If it is regression model, the results of ``forecast`` and ``predict`` are the same.

        * If it is autoregression model, this method will make predictions using true values
          instead of predicted on a previous step.
          It can be useful for making in-sample forecasts.

        To understand how a particular model behaves look at its documentation.

        Parameters
        ----------
        ts:
            Dataset with features
        prediction_size:
            Number of last timestamps to leave after making prediction.
            Previous timestamps will be used as a context for models that require it.

        Returns
        -------
        :
            Dataset with predictions
        """
        pass


class PredictionIntervalContextIgnorantAbstractModel(AbstractModel):
    """Interface for models that support prediction intervals and don't need context for prediction."""

    @property
    def context_size(self) -> int:
        """Context size of the model. Determines how many history points do we ask to pass to the model.

        Zero for this model.
        """
        return 0

    @abstractmethod
    def forecast(
        self, ts: TSDataset, prediction_interval: bool = False, quantiles: Sequence[float] = (0.025, 0.975)
    ) -> TSDataset:
        """Make predictions.

        * If it is regression model, the results of ``forecast`` and ``predict`` are the same.

        * If it is autoregression model, this method will make predictions using true values
          instead of predicted on a previous step.
          It can be useful for making in-sample forecasts.

        To understand how a particular model behaves look at its documentation.

        Parameters
        ----------
        ts:
            Dataset with features
        prediction_interval:
            If True returns prediction interval for forecast
        quantiles:
            Levels of prediction distribution. By default 2.5% and 97.5% are taken to form a 95% prediction interval

        Returns
        -------
        :
            Dataset with predictions
        """
        pass

    @abstractmethod
    def predict(
        self, ts: TSDataset, prediction_interval: bool = False, quantiles: Sequence[float] = (0.025, 0.975)
    ) -> TSDataset:
        """Make predictions.

        * If it is regression model, the results of ``forecast`` and ``predict`` are the same.

        * If it is autoregression model, this method will make predictions using true values
          instead of predicted on a previous step.
          It can be useful for making in-sample forecasts.

        To understand how a particular model behaves look at its documentation.

        Parameters
        ----------
        ts:
            Dataset with features
        prediction_interval:
            If True returns prediction interval for forecast
        quantiles:
            Levels of prediction distribution. By default 2.5% and 97.5% are taken to form a 95% prediction interval

        Returns
        -------
        :
            Dataset with predictions
        """
        pass


class PredictionIntervalContextRequiredAbstractModel(AbstractModel):
    """Interface for models that support prediction intervals and need context for prediction."""

    @abstractmethod
    def forecast(
        self,
        ts: TSDataset,
        prediction_size: int,
        prediction_interval: bool = False,
        quantiles: Sequence[float] = (0.025, 0.975),
    ) -> TSDataset:
        """Make predictions.

        * If it is regression model, the results of ``forecast`` and ``predict`` are the same.

        * If it is autoregression model, this method will make autoregressive predictions.

        To understand how a particular model behaves look at its documentation.

        Parameters
        ----------
        ts:
            Dataset with features
        prediction_size:
            Number of last timestamps to leave after making prediction.
            Previous timestamps will be used as a context for models that require it.
        prediction_interval:
            If True returns prediction interval for forecast
        quantiles:
            Levels of prediction distribution. By default 2.5% and 97.5% are taken to form a 95% prediction interval

        Returns
        -------
        :
            Dataset with predictions
        """
        pass

    @abstractmethod
    def predict(
        self,
        ts: TSDataset,
        prediction_size: int,
        prediction_interval: bool = False,
        quantiles: Sequence[float] = (0.025, 0.975),
    ) -> TSDataset:
        """Make predictions.

        * If it is regression model, the results of ``forecast`` and ``predict`` are the same.

        * If it is autoregression model, this method will make predictions using true values
          instead of predicted on a previous step.
          It can be useful for making in-sample forecasts.

        To understand how a particular model behaves look at its documentation.

        Parameters
        ----------
        ts:
            Dataset with features
        prediction_size:
            Number of last timestamps to leave after making prediction.
            Previous timestamps will be used as a context for models that require it.
        prediction_interval:
            If True returns prediction interval for forecast
        quantiles:
            Levels of prediction distribution. By default 2.5% and 97.5% are taken to form a 95% prediction interval

        Returns
        -------
        :
            Dataset with predictions
        """
        pass


class ModelForecastingMixin(ABC):
    """Base class for model mixins."""

    @abstractmethod
    def _forecast(self, **kwargs) -> TSDataset:
        pass

    @abstractmethod
    def _predict(self, **kwargs) -> TSDataset:
        pass


class NonPredictionIntervalContextIgnorantModelMixin(ModelForecastingMixin):
    """Mixin for models that don't support prediction intervals and don't need context for prediction."""

    def forecast(self, ts: TSDataset) -> TSDataset:
        """Make predictions.

        * If it is regression model, the results of ``forecast`` and ``predict`` are the same.

        * If it is autoregression model, this method will make autoregressive predictions.

        To understand how a particular model behaves look at its documentation.

        Parameters
        ----------
        ts:
            Dataset with features

        Returns
        -------
        :
            Dataset with predictions
        """
        return self._forecast(ts=ts)

    def predict(self, ts: TSDataset) -> TSDataset:
        """Make predictions.

        * If it is regression model, the results of ``forecast`` and ``predict`` are the same.

        * If it is autoregression model, this method will make predictions using true values
          instead of predicted on a previous step.
          It can be useful for making in-sample forecasts.

        To understand how a particular model behaves look at its documentation.

        Parameters
        ----------
        ts:
            Dataset with features

        Returns
        -------
        :
            Dataset with predictions
        """
        return self._predict(ts=ts)


class NonPredictionIntervalContextRequiredModelMixin(ModelForecastingMixin):
    """Mixin for models that don't support prediction intervals and need context for prediction."""

    def forecast(self, ts: TSDataset, prediction_size: int) -> TSDataset:
        """Make predictions.

        * If it is regression model, the results of ``forecast`` and ``predict`` are the same.

        * If it is autoregression model, this method will make autoregressive predictions.

        To understand how a particular model behaves look at its documentation.

        Parameters
        ----------
        ts:
            Dataset with features
        prediction_size:
            Number of last timestamps to leave after making prediction.
            Previous timestamps will be used as a context for models that require it.

        Returns
        -------
        :
            Dataset with predictions
        """
        return self._forecast(ts=ts, prediction_size=prediction_size)

    def predict(self, ts: TSDataset, prediction_size: int) -> TSDataset:
        """Make predictions.

        * If it is regression model, the results of ``forecast`` and ``predict`` are the same.

        * If it is autoregression model, this method will make predictions using true values
          instead of predicted on a previous step.
          It can be useful for making in-sample forecasts.

        To understand how a particular model behaves look at its documentation.

        Parameters
        ----------
        ts:
            Dataset with features
        prediction_size:
            Number of last timestamps to leave after making prediction.
            Previous timestamps will be used as a context for models that require it.

        Returns
        -------
        :
            Dataset with predictions
        """
        return self._predict(ts=ts, prediction_size=prediction_size)


class PredictionIntervalContextIgnorantModelMixin(ModelForecastingMixin):
    """Mixin for models that support prediction intervals and don't need context for prediction."""

    def forecast(
        self, ts: TSDataset, prediction_interval: bool = False, quantiles: Sequence[float] = (0.025, 0.975)
    ) -> TSDataset:
        """Make predictions.

        * If it is regression model, the results of ``forecast`` and ``predict`` are the same.

        * If it is autoregression model, this method will make autoregressive predictions.

        To understand how a particular model behaves look at its documentation.

        Parameters
        ----------
        ts:
            Dataset with features
        prediction_interval:
            If True returns prediction interval for forecast
        quantiles:
            Levels of prediction distribution. By default 2.5% and 97.5% are taken to form a 95% prediction interval

        Returns
        -------
        :
            Dataset with predictions
        """
        return self._forecast(ts=ts, prediction_interval=prediction_interval, quantiles=quantiles)

    def predict(
        self, ts: TSDataset, prediction_interval: bool = False, quantiles: Sequence[float] = (0.025, 0.975)
    ) -> TSDataset:
        """Make predictions.

        * If it is regression model, the results of ``forecast`` and ``predict`` are the same.

        * If it is autoregression model, this method will make predictions using true values
          instead of predicted on a previous step.
          It can be useful for making in-sample forecasts.

        To understand how a particular model behaves look at its documentation.

        Parameters
        ----------
        ts:
            Dataset with features
        prediction_interval:
            If True returns prediction interval for forecast
        quantiles:
            Levels of prediction distribution. By default 2.5% and 97.5% are taken to form a 95% prediction interval

        Returns
        -------
        :
            Dataset with predictions
        """
        return self._predict(ts=ts, prediction_interval=prediction_interval, quantiles=quantiles)


class PredictionIntervalContextRequiredModelMixin(ModelForecastingMixin):
    """Mixin for models that support prediction intervals and need context for prediction."""

    def forecast(
        self,
        ts: TSDataset,
        prediction_size: int,
        prediction_interval: bool = False,
        quantiles: Sequence[float] = (0.025, 0.975),
    ) -> TSDataset:
        """Make predictions.

        * If it is regression model, the results of ``forecast`` and ``predict`` are the same.

        * If it is autoregression model, this method will make autoregressive predictions.

        To understand how a particular model behaves look at its documentation.

        Parameters
        ----------
        ts:
            Dataset with features
        prediction_size:
            Number of last timestamps to leave after making prediction.
            Previous timestamps will be used as a context for models that require it.
        prediction_interval:
            If True returns prediction interval for forecast
        quantiles:
            Levels of prediction distribution. By default 2.5% and 97.5% are taken to form a 95% prediction interval

        Returns
        -------
        :
            Dataset with predictions
        """
        return self._forecast(
            ts=ts, prediction_size=prediction_size, prediction_interval=prediction_interval, quantiles=quantiles
        )

    def predict(
        self,
        ts: TSDataset,
        prediction_size: int,
        prediction_interval: bool = False,
        quantiles: Sequence[float] = (0.025, 0.975),
    ) -> TSDataset:
        """Make predictions.

        * If it is regression model, the results of ``forecast`` and ``predict`` are the same.

        * If it is autoregression model, this method will make predictions using true values
          instead of predicted on a previous step.
          It can be useful for making in-sample forecasts.

        To understand how a particular model behaves look at its documentation.

        Parameters
        ----------
        ts:
            Dataset with features
        prediction_size:
            Number of last timestamps to leave after making prediction.
            Previous timestamps will be used as a context for models that require it.
        prediction_interval:
            If True returns prediction interval for forecast
        quantiles:
            Levels of prediction distribution. By default 2.5% and 97.5% are taken to form a 95% prediction interval

        Returns
        -------
        :
            Dataset with predictions
        """
        return self._predict(
            ts=ts, prediction_size=prediction_size, prediction_interval=prediction_interval, quantiles=quantiles
        )


class PerSegmentModelMixin(ModelForecastingMixin):
    """Mixin for holding methods for per-segment prediction."""

    def __init__(self, base_model: Any):
        """
        Init PerSegmentModelMixin.

        Parameters
        ----------
        base_model:
            Internal model which will be used to forecast segments, expected to have fit/predict interface
        """
        self._base_model = base_model
        self._models: Optional[Dict[str, Any]] = None

    @log_decorator
    def fit(self, ts: TSDataset) -> "PerSegmentModelMixin":
        """Fit model.

        Parameters
        ----------
        ts:
            Dataset with features

        Returns
        -------
        :
            Model after fit
        """
        self._models = {}
        for segment in ts.segments:
            self._models[segment] = deepcopy(self._base_model)

        for segment, model in self._models.items():
            segment_features = ts[:, segment, :]
            segment_features = segment_features.dropna()  # TODO: https://github.com/tinkoff-ai/etna/issues/557
            segment_features = segment_features.droplevel("segment", axis=1)
            segment_features = segment_features.reset_index()
            model.fit(df=segment_features, regressors=ts.regressors)
        return self

    def _get_model(self) -> Dict[str, Any]:
        """Get internal etna base models that are used inside etna class.

        Returns
        -------
        :
           dictionary where key is segment and value is internal model
        """
        if self._models is None:
            raise ValueError("Can not get the dict with base models, the model is not fitted!")
        return self._models

    def get_model(self) -> Dict[str, Any]:
        """Get internal models that are used inside etna class.

        Internal model is a model that is used inside etna to forecast segments,
        e.g. :py:class:`catboost.CatBoostRegressor` or :py:class:`sklearn.linear_model.Ridge`.

        Returns
        -------
        :
           dictionary where key is segment and value is internal model
        """
        internal_models = {}
        for segment, base_model in self._get_model().items():
            if not hasattr(base_model, "get_model"):
                raise NotImplementedError(
                    f"get_model method is not implemented for {self._base_model.__class__.__name__}"
                )
            internal_models[segment] = base_model.get_model()
        return internal_models

    @staticmethod
    def _make_predictions_segment(
        model: Any, segment: str, ts: TSDataset, prediction_method: Callable, *args, **kwargs
    ) -> pd.DataFrame:
        """Make predictions for one segment."""
        segment_features = ts[:, segment, :]
        segment_features = segment_features.droplevel("segment", axis=1)
        segment_features = segment_features.reset_index()
        dates = segment_features["timestamp"]
        dates.reset_index(drop=True, inplace=True)
        segment_predict = prediction_method(self=model, df=segment_features, *args, **kwargs)
        if isinstance(segment_predict, np.ndarray):
            segment_predict = pd.DataFrame({"target": segment_predict})
        segment_predict["segment"] = segment

        prediction_size = kwargs.get("prediction_size")
        if prediction_size is not None:
            segment_predict["timestamp"] = dates[-prediction_size:].reset_index(drop=True)
        else:
            segment_predict["timestamp"] = dates
        return segment_predict

    def _make_predictions(self, ts: TSDataset, prediction_method: Callable, **kwargs) -> TSDataset:
        """Make predictions.

        Parameters
        ----------
        ts:
            Dataframe with features
        prediction_method:
            Method for making predictions

        Returns
        -------
        :
            Dataset with predictions
        """
        result_list = list()
        for segment, model in self._get_model().items():
            segment_predict = self._make_predictions_segment(
                model=model, segment=segment, ts=ts, prediction_method=prediction_method, **kwargs
            )

            result_list.append(segment_predict)

        result_df = pd.concat(result_list, ignore_index=True)
        result_df = result_df.set_index(["timestamp", "segment"])
        df = ts.to_pandas(flatten=True)
        df = df.set_index(["timestamp", "segment"])
        # clear values to be filled, otherwise during in-sample prediction new values won't be set
        columns_to_clear = result_df.columns.intersection(df.columns)
        df.loc[result_df.index, columns_to_clear] = np.NaN
        df = df.combine_first(result_df).reset_index()

        df = TSDataset.to_dataset(df)
        ts.df = df
        ts.inverse_transform()

        prediction_size = kwargs.get("prediction_size")
        if prediction_size is not None:
            ts.df = ts.df.iloc[-prediction_size:]
        return ts

    @log_decorator
    def _forecast(self, ts: TSDataset, **kwargs) -> TSDataset:
        if hasattr(self._base_model, "forecast"):
            return self._make_predictions(ts=ts, prediction_method=self._base_model.__class__.forecast, **kwargs)
        return self._make_predictions(ts=ts, prediction_method=self._base_model.__class__.predict, **kwargs)

    @log_decorator
    def _predict(self, ts: TSDataset, **kwargs) -> TSDataset:
        return self._make_predictions(ts=ts, prediction_method=self._base_model.__class__.predict, **kwargs)


class MultiSegmentModelMixin(ModelForecastingMixin):
    """Mixin for holding methods for multi-segment prediction.

    It currently isn't working with prediction intervals and context.
    """

    def __init__(self, base_model: Any):
        """
        Init MultiSegmentModel.

        Parameters
        ----------
        base_model:
            Internal model which will be used to forecast segments, expected to have fit/predict interface
        """
        self._base_model = base_model

    @log_decorator
    def fit(self, ts: TSDataset) -> "MultiSegmentModelMixin":
        """Fit model.

        Parameters
        ----------
        ts:
            Dataset with features

        Returns
        -------
        :
            Model after fit
        """
        df = ts.to_pandas(flatten=True)
        df = df.dropna()  # TODO: https://github.com/tinkoff-ai/etna/issues/557
        df = df.drop(columns="segment")
        self._base_model.fit(df=df, regressors=ts.regressors)
        return self

    def _make_predictions(self, ts: TSDataset, prediction_method: Callable, **kwargs) -> TSDataset:
        """Make predictions.

        Parameters
        ----------
        ts:
            Dataset with features
        prediction_method:
            Method for making predictions

        Returns
        -------
        :
            Dataset with predictions
        """
        horizon = len(ts.df)
        x = ts.to_pandas(flatten=True).drop(["segment"], axis=1)
        # TODO: make it work with prediction intervals and context
        y = prediction_method(self=self._base_model, ts=x, **kwargs).reshape(-1, horizon).T
        ts.loc[:, pd.IndexSlice[:, "target"]] = y
        ts.inverse_transform()
        return ts

    @log_decorator
    def _forecast(self, ts: TSDataset, **kwargs) -> TSDataset:
        if hasattr(self._base_model, "forecast"):
            return self._make_predictions(ts=ts, prediction_method=self._base_model.__class__.forecast, **kwargs)
        return self._make_predictions(ts=ts, prediction_method=self._base_model.__class__.predict, **kwargs)

    @log_decorator
    def _predict(self, ts: TSDataset, **kwargs) -> TSDataset:
        return self._make_predictions(ts=ts, prediction_method=self._base_model.__class__.predict, **kwargs)

    def get_model(self) -> Any:
        """Get internal model that is used inside etna class.

        Internal model is a model that is used inside etna to forecast segments,
        e.g. :py:class:`catboost.CatBoostRegressor` or :py:class:`sklearn.linear_model.Ridge`.

        Returns
        -------
        :
           Internal model
        """
        if not hasattr(self._base_model, "get_model"):
            raise NotImplementedError(f"get_model method is not implemented for {self._base_model.__class__.__name__}")
        return self._base_model.get_model()


class BaseAdapter(ABC):
    """Base class for models adapter."""

    @abstractmethod
    def get_model(self) -> Any:
        """Get internal model that is used inside etna class.

        Internal model is a model that is used inside etna to forecast segments,
        e.g. :py:class:`catboost.CatBoostRegressor` or :py:class:`sklearn.linear_model.Ridge`.

        Returns
        -------
        :
           Internal model
        """
        pass


class DeepAbstractNet(ABC):
    """Interface for etna native deep models."""

    @abstractmethod
    def make_samples(self, df: pd.DataFrame, encoder_length: int, decoder_length: int) -> Iterable[dict]:
        """Make samples from input slice of TSDataset.

        Parameters
        ----------
        df:
            slice is per-segment Dataframes
        encoder_length:
            encoder_length
        decoder_length:
            decoder_length

        Returns
        -------
        :
            samples of input slices
        """
        pass

    @abstractmethod
    def step(self, batch: dict, *args, **kwargs) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        """Make batch step.

        Parameters
        ----------
        batch:
            Batch with data to make inference on.

        Returns
        -------
        :
            loss, true_target, prediction_target
        """
        pass


class DeepBaseAbstractModel(ABC):
    """Interface for holding class of etna native deep models."""

    @abstractmethod
    def raw_fit(self, torch_dataset: "Dataset") -> "DeepBaseAbstractModel":
        """Fit model with torch like Dataset.

        Parameters
        ----------
        torch_dataset:
            Samples with data to fit on.

        Returns
        -------
        :
            Trained Model
        """
        pass

    @abstractmethod
    def raw_predict(self, torch_dataset: "Dataset") -> Dict[Tuple[str, str], np.ndarray]:
        """Make inference on torch like Dataset.

        Parameters
        ----------
        torch_dataset:
            Samples with data to make inference on.

        Returns
        -------
        :
            Predictions for each segment
        """
        pass

    @abstractmethod
    def get_model(self) -> "DeepBaseNet":
        """Get model.

        Returns
        -------
        :
           Torch Module
        """
        pass


class DeepBaseNet(DeepAbstractNet, LightningModule):
    """Class for partially implemented LightningModule interface."""

    def __init__(self):
        """Init DeepBaseNet."""
        super().__init__()

    def training_step(self, batch: dict, *args, **kwargs):  # type: ignore
        """Training step.

        Parameters
        ----------
        batch:
            batch of data

        Returns
        -------
        :
            loss
        """
        loss, _, _ = self.step(batch, *args, **kwargs)  # type: ignore
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch: dict, *args, **kwargs):  # type: ignore
        """Validate step.

        Parameters
        ----------
        batch:
            batch of data

        Returns
        -------
        :
            loss
        """
        loss, _, _ = self.step(batch, *args, **kwargs)  # type: ignore
        self.log("val_loss", loss, on_epoch=True)
        return loss


class DeepBaseModel(DeepBaseAbstractModel, NonPredictionIntervalContextRequiredAbstractModel):
    """Class for partially implemented interfaces for holding deep models."""

    def __init__(
        self,
        *,
        net: DeepBaseNet,
        encoder_length: int,
        decoder_length: int,
        train_batch_size: int,
        test_batch_size: int,
        trainer_params: Optional[dict],
        train_dataloader_params: Optional[dict],
        test_dataloader_params: Optional[dict],
        val_dataloader_params: Optional[dict],
        split_params: Optional[dict],
    ):
        """Init DeepBaseModel.

        Parameters
        ----------
        net:
            network to train
        encoder_length:
            encoder length
        decoder_length:
            decoder length
        train_batch_size:
            batch size for training
        test_batch_size:
            batch size for testing
        trainer_params:
            Pytorch ligthning trainer parameters (api reference :py:class:`pytorch_lightning.trainer.trainer.Trainer`)
        train_dataloader_params:
            parameters for train dataloader like sampler for example (api reference :py:class:`torch.utils.data.DataLoader`)
        test_dataloader_params:
            parameters for test dataloader
        val_dataloader_params:
            parameters for validation dataloader
        split_params:
            dictionary with parameters for :py:func:`torch.utils.data.random_split` for train-test splitting
                * **train_size**: (*float*) value from 0 to 1 - fraction of samples to use for training

                * **generator**: (*Optional[torch.Generator]*) - generator for reproducibile train-test splitting

                * **torch_dataset_size**: (*Optional[int]*) - number of samples in dataset, in case of dataset not implementing ``__len__``
        """
        super().__init__()
        self.net = net
        self.encoder_length = encoder_length
        self.decoder_length = decoder_length
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.train_dataloader_params = {} if train_dataloader_params is None else train_dataloader_params
        self.test_dataloader_params = {} if test_dataloader_params is None else test_dataloader_params
        self.val_dataloader_params = {} if val_dataloader_params is None else val_dataloader_params
        self.trainer_params = {} if trainer_params is None else trainer_params
        self.split_params = {} if split_params is None else split_params

    @property
    def context_size(self) -> int:
        """Context size of the model."""
        return self.encoder_length

    @log_decorator
    def fit(self, ts: TSDataset) -> "DeepBaseModel":
        """Fit model.

        Parameters
        ----------
        ts:
            TSDataset with features

        Returns
        -------
        :
            Model after fit
        """
        torch_dataset = ts.to_torch_dataset(
            functools.partial(
                self.net.make_samples, encoder_length=self.encoder_length, decoder_length=self.decoder_length
            ),
            dropna=True,
        )
        self.raw_fit(torch_dataset)
        return self

    def raw_fit(self, torch_dataset: "Dataset") -> "DeepBaseModel":
        """Fit model on torch like Dataset.

        Parameters
        ----------
        torch_dataset:
            Torch like dataset for model fit

        Returns
        -------
        :
            Model after fit
        """
        if self.split_params:
            if isinstance(torch_dataset, Sized):
                torch_dataset_size = len(torch_dataset)
            else:
                torch_dataset_size = self.split_params["torch_dataset_size"]
            if torch_dataset_size is None:
                raise ValueError("torch_dataset_size must be provided if torch_dataset is not sized")
            train_size = self.split_params["train_size"]
            torch_dataset_train_size = int(torch_dataset_size * train_size)
            torch_dataset_val_size = torch_dataset_size - torch_dataset_train_size
            train_dataset, val_dataset = random_split(
                torch_dataset,
                lengths=[
                    torch_dataset_train_size,
                    torch_dataset_val_size,
                ],
                generator=self.split_params.get("generator"),
            )
            train_dataloader = DataLoader(
                train_dataset, batch_size=self.train_batch_size, shuffle=True, **self.train_dataloader_params
            )
            val_dataloader: Optional[DataLoader] = DataLoader(
                val_dataset, batch_size=self.test_batch_size, shuffle=False, **self.val_dataloader_params
            )
        else:
            train_dataloader = DataLoader(
                torch_dataset, batch_size=self.train_batch_size, shuffle=True, **self.train_dataloader_params
            )
            val_dataloader = None

        if "logger" not in self.trainer_params:
            self.trainer_params["logger"] = tslogger.pl_loggers
        else:
            self.trainer_params["logger"] += tslogger.pl_loggers

        trainer = Trainer(**self.trainer_params)
        trainer.fit(self.net, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
        return self

    def raw_predict(self, torch_dataset: "Dataset") -> Dict[Tuple[str, str], np.ndarray]:
        """Make inference on torch like Dataset.

        Parameters
        ----------
        torch_dataset:
            Torch like dataset for model inference

        Returns
        -------
        :
            Dictionary with predictions
        """
        test_dataloader = DataLoader(
            torch_dataset, batch_size=self.test_batch_size, shuffle=False, **self.test_dataloader_params
        )

        predictions_dict = dict()
        self.net.eval()
        with torch.no_grad():
            for batch in test_dataloader:
                segments = batch["segment"]
                predictions = self.net(batch)
                predictions_array = predictions.numpy()
                for idx, segment in enumerate(segments):
                    predictions_dict[(segment, "target")] = predictions_array[
                        idx, :
                    ]  # TODO: rethink in case of issue-791

        return predictions_dict

    @log_decorator
    def forecast(self, ts: "TSDataset", prediction_size: int) -> "TSDataset":
        """Make predictions.

        This method will make autoregressive predictions.

        Parameters
        ----------
        ts:
            Dataset with features and expected decoder length for context
        prediction_size:
            Number of last timestamps to leave after making prediction.
            Previous timestamps will be used as a context.

        Returns
        -------
        :
            Dataset with predictions
        """
        test_dataset = ts.to_torch_dataset(
            make_samples=functools.partial(
                self.net.make_samples, encoder_length=self.encoder_length, decoder_length=self.decoder_length
            ),
            dropna=False,
        )
        predictions = self.raw_predict(test_dataset)
        future_ts = ts.tsdataset_idx_slice(start_idx=self.encoder_length, end_idx=self.encoder_length + prediction_size)
        for (segment, feature_nm), value in predictions.items():
            future_ts.df.loc[:, pd.IndexSlice[segment, feature_nm]] = value[:prediction_size, :]

        future_ts.inverse_transform()

        return future_ts

    @log_decorator
    def predict(self, ts: "TSDataset", prediction_size: int) -> "TSDataset":
        """Make predictions.

        This method will make predictions using true values instead of predicted on a previous step.
        It can be useful for making in-sample forecasts.

        Parameters
        ----------
        ts:
            Dataset with features and expected decoder length for context
        prediction_size:
            Number of last timestamps to leave after making prediction.
            Previous timestamps will be used as a context.

        Returns
        -------
        :
            Dataset with predictions
        """
        raise NotImplementedError("It is currently not implemented!")

    def get_model(self) -> "DeepBaseNet":
        """Get model.

        Returns
        -------
        :
           Torch Module
        """
        return self.net


ModelType = Union[
    NonPredictionIntervalContextIgnorantAbstractModel,
    NonPredictionIntervalContextRequiredAbstractModel,
    PredictionIntervalContextIgnorantAbstractModel,
    PredictionIntervalContextRequiredAbstractModel,
]

ContextRequiredModelType = Union[
    NonPredictionIntervalContextRequiredAbstractModel,
    PredictionIntervalContextRequiredAbstractModel,
]

ContextIgnorantModelType = Union[
    NonPredictionIntervalContextIgnorantAbstractModel,
    PredictionIntervalContextIgnorantAbstractModel,
]
