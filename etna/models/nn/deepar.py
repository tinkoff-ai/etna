from typing import Any
from typing import Dict
from typing import Optional
from typing import Sequence
from typing import Union

import pandas as pd

from etna import SETTINGS
from etna.datasets.tsdataset import TSDataset
from etna.models.base import MultiSegmentPredictionIntervalModel
from etna.models.base import log_decorator
from etna.models.nn.utils import PytorchForecastingDatasetBuilder
from etna.models.nn.utils import PytorchForecastingMixin
from etna.models.nn.utils import _DeepCopyMixin

if SETTINGS.torch_required:
    from pytorch_forecasting.data import GroupNormalizer
    from pytorch_forecasting.data import TimeSeriesDataSet
    from pytorch_forecasting.metrics import DistributionLoss
    from pytorch_forecasting.metrics import NormalDistributionLoss
    from pytorch_forecasting.models import DeepAR
    from pytorch_lightning import LightningModule


class DeepARModel(PytorchForecastingMixin, MultiSegmentPredictionIntervalModel, _DeepCopyMixin):
    """Wrapper for :py:class:`pytorch_forecasting.models.deepar.DeepAR`.

    Notes
    -----
    We save :py:class:`pytorch_forecasting.data.timeseries.TimeSeriesDataSet` in instance to use it in the model.
    It`s not right pattern of using Transforms and TSDataset.
    """

    def __init__(
        self,
        decoder_length: Optional[int] = None,
        encoder_length: Optional[int] = None,
        dataset_builder: Optional[PytorchForecastingDatasetBuilder] = None,
        train_batch_size: int = 64,
        test_batch_size: int = 64,
        lr: Optional[float] = 1e-3,
        cell_type: str = "LSTM",
        hidden_size: int = 10,
        rnn_layers: int = 2,
        dropout: float = 0.1,
        loss: Optional["DistributionLoss"] = None,
        trainer_params: Optional[Dict[str, Any]] = None,
        quantiles_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize DeepAR wrapper.

        Parameters
        ----------
        decoder_length:
            Decoder length.
        encoder_length:
            Encoder length.
        dataset_builder:
            Dataset builder for PytorchForecasting.
        train_batch_size:
            Train batch size.
        test_batch_size:
            Test  batch size.
        lr:
            Learning rate.
        cell_type:
            One of 'LSTM', 'GRU'.
        hidden_size:
            Hidden size of network which can range from 8 to 512.
        rnn_layers:
            Number of LSTM layers.
        dropout:
            Dropout rate.
        loss:
            Distribution loss function. Keep in mind that each distribution
            loss function might have specific requirements for target normalization.
            Defaults to :py:class:`pytorch_forecasting.metrics.NormalDistributionLoss`.
        trainer_params:
            Additional arguments for pytorch_lightning Trainer.
        quantiles_kwargs:
            Additional arguments for computing quantiles, look at ``to_quantiles()`` method for your loss.
        """
        super().__init__()
        if loss is None:
            loss = NormalDistributionLoss()

        if (encoder_length is None or decoder_length is None) and dataset_builder is not None:

            self.encoder_length = dataset_builder.max_encoder_length
            self.decoder_length = dataset_builder.max_prediction_length
            self.dataset_builder = dataset_builder
        elif (encoder_length is not None and decoder_length is not None) and dataset_builder is None:
            self.encoder_length = encoder_length
            self.decoder_length = decoder_length
            self.dataset_builder = PytorchForecastingDatasetBuilder(
                max_encoder_length=encoder_length,
                min_encoder_length=encoder_length,
                max_prediction_length=decoder_length,
                time_varying_known_reals=["time_idx"],
                time_varying_unknown_reals=["target"],
                target_normalizer=GroupNormalizer(groups=["segment"]),
            )
        else:
            raise ValueError("You should provide either dataset_builder or encoder_length and decoder_length")

        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.lr = lr
        self.cell_type = cell_type
        self.hidden_size = hidden_size
        self.rnn_layers = rnn_layers
        self.dropout = dropout
        self.loss = loss
        self.trainer_params = trainer_params if trainer_params is not None else dict()
        self.quantiles_kwargs = quantiles_kwargs if quantiles_kwargs is not None else dict()
        self.model: Optional[Union[LightningModule, DeepAR]] = None
        self._last_train_timestamp = None

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
            learning_rate=[self.lr],
            cell_type=self.cell_type,
            hidden_size=self.hidden_size,
            rnn_layers=self.rnn_layers,
            dropout=self.dropout,
            loss=self.loss,
        )

    @log_decorator
    def forecast(
        self,
        ts: TSDataset,
        horizon: int,
        prediction_interval: bool = False,
        quantiles: Sequence[float] = (0.025, 0.975),
    ) -> TSDataset:
        """
        Predict future.

        Parameters
        ----------
        ts:
            Dataset with features
        horizon:
            Forecasting horizon
        prediction_interval:
            If True returns prediction interval for forecast
        quantiles:
            Levels of prediction distribution. By default 2.5% and 97.5% are taken to form a 95% prediction interval

        Returns
        -------
        TSDataset
            TSDataset with predictions.
        """
        ts, prediction_dataloader = self._make_target_prediction(ts, horizon)

        if prediction_interval:
            quantiles_predicts = self.model.predict(  # type: ignore
                prediction_dataloader,
                mode="quantiles",
                mode_kwargs={"quantiles": quantiles, **self.quantiles_kwargs},
            ).numpy()
            # shape (segments, encoder_length, len(quantiles))
            quantiles_predicts = quantiles_predicts.transpose((1, 0, 2))
            # shape (encoder_length, segments, len(quantiles))
            quantiles_predicts = quantiles_predicts.reshape(quantiles_predicts.shape[0], -1)
            # shape (encoder_length, segments * len(quantiles))

            df = ts.df
            segments = ts.segments
            quantile_columns = [f"target_{quantile:.4g}" for quantile in quantiles]
            columns = pd.MultiIndex.from_product([segments, quantile_columns])
            quantiles_df = pd.DataFrame(quantiles_predicts[: len(df)], columns=columns, index=df.index)
            df = pd.concat((df, quantiles_df), axis=1)
            df = df.sort_index(axis=1)
            ts.df = df

        return ts
