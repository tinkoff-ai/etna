import functools
import inspect
from abc import ABC
from abc import abstractmethod
from copy import deepcopy
from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
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
        :
            Model after fit
        """
        pass

    @abstractmethod
    def forecast(
        self, ts: TSDataset, prediction_interval: bool = False, quantiles: Sequence[float] = (0.025, 0.975)
    ) -> TSDataset:
        """Make predictions.

        Parameters
        ----------
        ts:
            Dataframe with features
        prediction_interval:
            If True returns prediction interval for forecast
        quantiles:
            Levels of prediction distribution. By default 2.5% and 97.5% taken to form a 95% prediction interval

        Returns
        -------
        TSDataset
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


class FitAbstractModel(ABC):
    """Interface for model with fit method."""

    @abstractmethod
    def fit(self, ts: TSDataset) -> "FitAbstractModel":
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


class ForecastAbstractModel(ABC):
    """Interface for model with forecast method."""

    @abstractmethod
    def forecast(self, ts: TSDataset) -> TSDataset:
        """Make predictions.

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


class PredictIntervalAbstractModel(ABC):
    """Interface for model with forecast method that creates prediction interval."""

    @abstractmethod
    def forecast(
        self, ts: TSDataset, prediction_interval: bool = False, quantiles: Sequence[float] = (0.025, 0.975)
    ) -> TSDataset:
        """Make predictions.

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


class PerSegmentBaseModel(FitAbstractModel, BaseMixin):
    """Base class for holding specific models for per-segment prediction."""

    def __init__(self, base_model: Any):
        """
        Init PerSegmentBaseModel.

        Parameters
        ----------
        base_model:
            Internal model which will be used to forecast segments, expected to have fit/predict interface
        """
        self._base_model = base_model
        self._models: Optional[Dict[str, Any]] = None

    @log_decorator
    def fit(self, ts: TSDataset) -> "PerSegmentBaseModel":
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
    def _forecast_segment(model: Any, segment: str, ts: TSDataset, *args, **kwargs) -> pd.DataFrame:
        """Make predictions for one segment."""
        segment_features = ts[:, segment, :]
        segment_features = segment_features.droplevel("segment", axis=1)
        segment_features = segment_features.reset_index()
        dates = segment_features["timestamp"]
        dates.reset_index(drop=True, inplace=True)
        segment_predict = model.predict(df=segment_features, *args, **kwargs)
        if isinstance(segment_predict, np.ndarray):
            segment_predict = pd.DataFrame({"target": segment_predict})
        segment_predict["segment"] = segment
        segment_predict["timestamp"] = dates
        return segment_predict


class PerSegmentModel(PerSegmentBaseModel, ForecastAbstractModel):
    """Class for holding specific models for per-segment prediction."""

    def __init__(self, base_model: Any):
        """
        Init PerSegmentBaseModel.

        Parameters
        ----------
        base_model:
            Internal model which will be used to forecast segments, expected to have fit/predict interface
        """
        super().__init__(base_model=base_model)

    @log_decorator
    def forecast(self, ts: TSDataset) -> TSDataset:
        """Make predictions.

        Parameters
        ----------
        ts:
            Dataframe with features
        Returns
        -------
        :
            Dataset with predictions
        """
        result_list = list()
        for segment, model in self._get_model().items():
            segment_predict = self._forecast_segment(model=model, segment=segment, ts=ts)
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


class PerSegmentPredictionIntervalModel(PerSegmentBaseModel, PredictIntervalAbstractModel):
    """Class for holding specific models for per-segment prediction which are able to build prediction intervals."""

    def __init__(self, base_model: Any):
        """
        Init PerSegmentPredictionIntervalModel.

        Parameters
        ----------
        base_model:
            Internal model which will be used to forecast segments, expected to have fit/predict interface
        """
        super().__init__(base_model=base_model)

    @log_decorator
    def forecast(
        self, ts: TSDataset, prediction_interval: bool = False, quantiles: Sequence[float] = (0.025, 0.975)
    ) -> TSDataset:
        """Make predictions.

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
        result_list = list()
        for segment, model in self._get_model().items():
            segment_predict = self._forecast_segment(
                model=model, segment=segment, ts=ts, prediction_interval=prediction_interval, quantiles=quantiles
            )
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


class MultiSegmentModel(FitAbstractModel, ForecastAbstractModel, BaseMixin):
    """Class for holding specific models for per-segment prediction."""

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
    def fit(self, ts: TSDataset) -> "MultiSegmentModel":
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

    @log_decorator
    def forecast(self, ts: TSDataset) -> TSDataset:
        """Make predictions.

        Parameters
        ----------
        ts:
            Dataset with features

        Returns
        -------
        :
            Dataset with predictions
        """
        horizon = len(ts.df)
        x = ts.to_pandas(flatten=True).drop(["segment"], axis=1)
        y = self._base_model.predict(x).reshape(-1, horizon).T
        ts.loc[:, pd.IndexSlice[:, "target"]] = y
        ts.inverse_transform()
        return ts

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
    def forecast(self, ts: TSDataset, horizon: int) -> TSDataset:
        """Make predictions.

        Parameters
        ----------
        ts:
            Dataset with features and expected decoder length for context
        horizon:
            Horizon to predict for

        Returns
        -------
        :
            Dataset with predictions
        """
        pass

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


class DeepBaseModel(FitAbstractModel, DeepBaseAbstractModel, BaseMixin):
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
    def forecast(self, ts: "TSDataset", horizon: int) -> "TSDataset":
        """Make predictions.

        Parameters
        ----------
        ts:
            Dataset with features and expected decoder length for context
        horizon:
            Horizon to predict for

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
        future_ts = ts.tsdataset_idx_slice(start_idx=self.encoder_length, end_idx=self.encoder_length + horizon)
        for (segment, feature_nm), value in predictions.items():
            future_ts.df.loc[:, pd.IndexSlice[segment, feature_nm]] = value[:horizon, :]

        future_ts.inverse_transform()

        return future_ts

    def get_model(self) -> "DeepBaseNet":
        """Get model.

        Returns
        -------
        :
           Torch Module
        """
        return self.net


class MultiSegmentPredictionIntervalModel(FitAbstractModel, PredictIntervalAbstractModel, BaseMixin):
    """Class for holding specific models for multi-segment prediction which are able to build prediction intervals."""

    def __init__(self):
        """Init MultiSegmentPredictionIntervalModel."""
        self.model = None

    def get_model(self) -> Any:
        """Get internal model that is used inside etna class.

        Internal model is a model that is used inside etna to forecast segments,
        e.g. :py:class:`catboost.CatBoostRegressor` or :py:class:`sklearn.linear_model.Ridge`.

        Returns
        -------
        :
           Internal model
        """
        return self.model


BaseModel = Union[
    PerSegmentModel,
    PerSegmentPredictionIntervalModel,
    MultiSegmentModel,
    DeepBaseModel,
    MultiSegmentPredictionIntervalModel,
]
