import functools
from abc import ABC
from abc import abstractmethod
from typing import Any
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
from etna.core import AbstractSaveable
from etna.core import SaveMixin
from etna.core.mixins import BaseMixin
from etna.datasets.tsdataset import TSDataset
from etna.loggers import tslogger
from etna.models.decorators import log_decorator
from etna.models.mixins import SaveNNMixin

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
    SaveNNMixin = Mock  # type: ignore


class AbstractModel(SaveMixin, AbstractSaveable, ABC, BaseMixin):
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
        """Make predictions with using true values as autoregression context if possible (teacher forcing).

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
        """Make predictions with using true values as autoregression context if possible (teacher forcing).

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
        """Make predictions with using true values as autoregression context if possible (teacher forcing).

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
        """Make predictions with using true values as autoregression context if possible (teacher forcing).

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


class DeepBaseModel(DeepBaseAbstractModel, SaveNNMixin, NonPredictionIntervalContextRequiredAbstractModel):
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
        self.trainer: Optional[Trainer] = None

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

        self.trainer = Trainer(**self.trainer_params)
        self.trainer.fit(self.net, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
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
                self.net.make_samples, encoder_length=self.encoder_length, decoder_length=prediction_size
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
        raise NotImplementedError("Method predict isn't currently implemented!")

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

PredictionIntervalModelType = Union[
    PredictionIntervalContextIgnorantAbstractModel, PredictionIntervalContextRequiredAbstractModel
]

NonPredictionIntervalModelType = Union[
    NonPredictionIntervalContextIgnorantAbstractModel, NonPredictionIntervalContextRequiredAbstractModel
]
