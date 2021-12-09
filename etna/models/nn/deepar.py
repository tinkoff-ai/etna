from typing import List
from typing import Optional
from typing import Union

import pandas as pd

from etna import SETTINGS
from etna.datasets.tsdataset import TSDataset
from etna.loggers import tslogger
from etna.models.base import Model
from etna.models.base import log_decorator
from etna.transforms import PytorchForecastingTransform

if SETTINGS.torch_required:
    import pytorch_lightning as pl
    from pytorch_forecasting.data import TimeSeriesDataSet
    from pytorch_forecasting.models import DeepAR
    from pytorch_lightning import LightningModule


class DeepARModel(Model):
    """Wrapper for DeepAR from Pytorch Forecasting library.

    Notes
    -----
    We save TimeSeriesDataSet in instance to use it in the model.
    It`s not right pattern of using Transforms and TSDataset.
    """

    def __init__(
        self,
        batch_size: int = 64,
        context_length: Optional[int] = None,
        max_epochs: int = 10,
        gpus: Union[int, List[int]] = 0,
        gradient_clip_val: float = 0.1,
        learning_rate: List[float] = [0.001],
        cell_type: str = "LSTM",
        hidden_size: int = 10,
        rnn_layers: int = 2,
        dropout: float = 0.1,
    ):
        """
        Initialize DeepAR wrapper.

        Parameters
        ----------
        batch_size:
            Batch size.
        context_length:
            Max encoder length, if None max encoder length is equal to 2 horizons.
        max_epochs:
            Max epochs.
        gpus:
            0 - is CPU, or [n_{i}] - to choose n_{i} GPU from cluster.
        gradient_clip_val:
            Clipping by norm is using, choose 0 to not clip.
        learning_rate:
            Learning rate.
        cell_type:
            One of 'LSTM', 'GRU'.
        hidden_size:
            Hidden size of network which can range from 8 to 512.
        rnn_layers:
            Number of LSTM layers.
        dropout:
            Dropout rate.
        """
        self.max_epochs = max_epochs
        self.gpus = gpus
        self.gradient_clip_val = gradient_clip_val
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.context_length = context_length
        self.cell_type = cell_type
        self.hidden_size = hidden_size
        self.rnn_layers = rnn_layers
        self.dropout = dropout

        self.model: Optional[Union[LightningModule, DeepAR]] = None
        self.trainer: Optional[pl.Trainer] = None

    def _from_dataset(self, ts_dataset: TimeSeriesDataSet) -> LightningModule:
        """
        Construct DeepAR.

        Returns
        -------
        DeepAR
            Class instance.
        """
        return DeepAR.from_dataset(
            ts_dataset,
            learning_rate=self.learning_rate,
            cell_type=self.cell_type,
            hidden_size=self.hidden_size,
            rnn_layers=self.rnn_layers,
            dropout=self.dropout,
        )

    @staticmethod
    def _get_pf_transform(ts: TSDataset) -> PytorchForecastingTransform:
        """Get PytorchForecastingTransform from ts.transforms or raise exception if not found."""
        if ts.transforms is not None and isinstance(ts.transforms[-1], PytorchForecastingTransform):
            return ts.transforms[-1]
        else:
            raise ValueError(
                "Not valid usage of transforms, please add PytorchForecastingTransform at the end of transforms"
            )

    @log_decorator
    def fit(self, ts: TSDataset) -> "DeepARModel":
        """
        Fit model.

        Parameters
        ----------
        ts:
            TSDataset to fit.

        Returns
        -------
        DeepARModel
        """
        pf_transform = self._get_pf_transform(ts)
        self.model = self._from_dataset(pf_transform.pf_dataset_train)

        self.trainer = pl.Trainer(
            logger=tslogger.pl_loggers,
            max_epochs=self.max_epochs,
            gpus=self.gpus,
            checkpoint_callback=False,
            gradient_clip_val=self.gradient_clip_val,
        )

        train_dataloader = pf_transform.pf_dataset_train.to_dataloader(train=True, batch_size=self.batch_size)

        self.trainer.fit(self.model, train_dataloader)

        return self

    @log_decorator
    def forecast(self, ts: TSDataset) -> TSDataset:
        """
        Predict future.

        Parameters
        ----------
        ts:
            TSDataset to forecast.

        Returns
        -------
        TSDataset
            TSDataset with predictions.
        """
        pf_transform = self._get_pf_transform(ts)
        prediction_dataloader = pf_transform.pf_dataset_predict.to_dataloader(
            train=False, batch_size=self.batch_size * 2
        )

        predicts = self.model.predict(prediction_dataloader).numpy()  # type: ignore
        # shape (segments, encoder_length)

        ts.loc[:, pd.IndexSlice[:, "target"]] = predicts.T[-len(ts.df) :]
        return ts
