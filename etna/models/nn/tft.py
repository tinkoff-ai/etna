import warnings
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Union

import pandas as pd

from etna import SETTINGS
from etna.datasets.tsdataset import TSDataset
from etna.loggers import tslogger
from etna.models.base import PredictionIntervalContextIgnorantAbstractModel
from etna.models.base import log_decorator
from etna.models.nn.utils import _DeepCopyMixin
from etna.transforms import PytorchForecastingTransform

if SETTINGS.torch_required:
    import pytorch_lightning as pl
    from pytorch_forecasting.data import TimeSeriesDataSet
    from pytorch_forecasting.metrics import MultiHorizonMetric
    from pytorch_forecasting.metrics import QuantileLoss
    from pytorch_forecasting.models import TemporalFusionTransformer
    from pytorch_lightning import LightningModule


class TFTModel(_DeepCopyMixin, PredictionIntervalContextIgnorantAbstractModel):
    """Wrapper for :py:class:`pytorch_forecasting.models.temporal_fusion_transformer.TemporalFusionTransformer`.

    Notes
    -----
    We save :py:class:`pytorch_forecasting.data.timeseries.TimeSeriesDataSet` in instance to use it in the model.
    It`s not right pattern of using Transforms and TSDataset.
    """

    context_size = 0

    def __init__(
        self,
        max_epochs: int = 10,
        gpus: Union[int, List[int]] = 0,
        gradient_clip_val: float = 0.1,
        learning_rate: Optional[List[float]] = None,
        batch_size: int = 64,
        context_length: Optional[int] = None,
        hidden_size: int = 16,
        lstm_layers: int = 1,
        attention_head_size: int = 4,
        dropout: float = 0.1,
        hidden_continuous_size: int = 8,
        loss: "MultiHorizonMetric" = None,
        trainer_kwargs: Optional[Dict[str, Any]] = None,
        quantiles_kwargs: Optional[Dict[str, Any]] = None,
        *args,
        **kwargs,
    ):
        """
        Initialize TFT wrapper.

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
        hidden_size:
            Hidden size of network which can range from 8 to 512.
        lstm_layers:
            Number of LSTM layers.
        attention_head_size:
            Number of attention heads.
        dropout:
            Dropout rate.
        hidden_continuous_size:
            Hidden size for processing continuous variables.
        loss:
            Loss function taking prediction and targets.
            Defaults to :py:class:`pytorch_forecasting.metrics.QuantileLoss`.
        trainer_kwargs:
            Additional arguments for pytorch_lightning Trainer.
        quantiles_kwargs:
            Additional arguments for computing quantiles, look at ``to_quantiles()`` method for your loss.
        """
        super().__init__()
        if loss is None:
            loss = QuantileLoss()
        self.max_epochs = max_epochs
        self.gpus = gpus
        self.gradient_clip_val = gradient_clip_val
        self.learning_rate = learning_rate if learning_rate is not None else [0.001]
        self.horizon = None
        self.batch_size = batch_size
        self.context_length = context_length
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.attention_head_size = attention_head_size
        self.dropout = dropout
        self.hidden_continuous_size = hidden_continuous_size
        self.loss = loss
        self.trainer_kwargs = trainer_kwargs if trainer_kwargs is not None else dict()
        self.quantiles_kwargs = quantiles_kwargs if quantiles_kwargs is not None else dict()
        self.model: Optional[Union[LightningModule, TemporalFusionTransformer]] = None
        self.trainer: Optional[pl.Trainer] = None
        self._last_train_timestamp = None
        self._freq: Optional[str] = None

    def _from_dataset(self, ts_dataset: TimeSeriesDataSet) -> LightningModule:
        """
        Construct TemporalFusionTransformer.

        Returns
        -------
        LightningModule class instance.
        """
        return TemporalFusionTransformer.from_dataset(
            ts_dataset,
            learning_rate=self.learning_rate,
            hidden_size=self.hidden_size,
            lstm_layers=self.lstm_layers,
            attention_head_size=self.attention_head_size,
            dropout=self.dropout,
            hidden_continuous_size=self.hidden_continuous_size,
            loss=self.loss,
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
    def fit(self, ts: TSDataset) -> "TFTModel":
        """
        Fit model.

        Parameters
        ----------
        ts:
            TSDataset to fit.

        Returns
        -------
        TFTModel
        """
        self._last_train_timestamp = ts.df.index[-1]
        self._freq = ts.freq
        pf_transform = self._get_pf_transform(ts)
        self.model = self._from_dataset(pf_transform.pf_dataset_train)

        trainer_kwargs = dict(
            logger=tslogger.pl_loggers,
            max_epochs=self.max_epochs,
            gpus=self.gpus,
            gradient_clip_val=self.gradient_clip_val,
        )
        trainer_kwargs.update(self.trainer_kwargs)

        self.trainer = pl.Trainer(**trainer_kwargs)

        train_dataloader = pf_transform.pf_dataset_train.to_dataloader(train=True, batch_size=self.batch_size)

        self.trainer.fit(self.model, train_dataloader)

        return self

    @log_decorator
    def forecast(
        self, ts: TSDataset, prediction_interval: bool = False, quantiles: Sequence[float] = (0.025, 0.975)
    ) -> TSDataset:
        """Make predictions.

        This method will make autoregressive predictions.

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
        TSDataset
            TSDataset with predictions.
        """
        if ts.index[0] <= self._last_train_timestamp:
            raise NotImplementedError(
                "It is not possible to make in-sample predictions with TFT model! "
                "In-sample predictions aren't supported by current implementation."
            )
        elif ts.index[0] != pd.date_range(self._last_train_timestamp, periods=2, freq=self._freq)[-1]:
            raise NotImplementedError(
                "You can only forecast from the next point after the last one in the training dataset: "
                f"last train timestamp: {self._last_train_timestamp}, first test timestamp is {ts.index[0]}"
            )
        else:
            pass

        pf_transform = self._get_pf_transform(ts)
        if pf_transform.pf_dataset_predict is None:
            raise ValueError(
                "The future is not generated! Generate future using TSDataset make_future before calling forecast method!"
            )
        prediction_dataloader = pf_transform.pf_dataset_predict.to_dataloader(
            train=False, batch_size=self.batch_size * 2
        )

        predicts = self.model.predict(prediction_dataloader).numpy()  # type: ignore
        # shape (segments, encoder_length)
        ts.loc[:, pd.IndexSlice[:, "target"]] = predicts.T[: len(ts.df)]

        if prediction_interval:
            if not isinstance(self.loss, QuantileLoss):
                warnings.warn(
                    "Quantiles can't be computed because TFTModel supports this only if QunatileLoss is chosen"
                )
            else:
                quantiles_predicts = self.model.predict(  # type: ignore
                    prediction_dataloader,
                    mode="quantiles",
                    mode_kwargs={"quantiles": quantiles, **self.quantiles_kwargs},
                ).numpy()
                # shape (segments, encoder_length, len(quantiles))

                loss_quantiles = self.loss.quantiles
                computed_quantiles_indices = []
                computed_quantiles = []
                not_computed_quantiles = []
                for quantile in quantiles:
                    if quantile in loss_quantiles:
                        computed_quantiles.append(quantile)
                        computed_quantiles_indices.append(loss_quantiles.index(quantile))
                    else:
                        not_computed_quantiles.append(quantile)

                if not_computed_quantiles:
                    warnings.warn(
                        f"Quantiles: {not_computed_quantiles} can't be computed because loss wasn't fitted on them"
                    )

                quantiles_predicts = quantiles_predicts[:, :, computed_quantiles_indices]
                quantiles = computed_quantiles

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

        ts.inverse_transform()
        return ts

    @log_decorator
    def predict(
        self, ts: TSDataset, prediction_interval: bool = False, quantiles: Sequence[float] = (0.025, 0.975)
    ) -> TSDataset:
        """Make predictions.

        This method will make predictions using true values instead of predicted on a previous step.
        It can be useful for making in-sample forecasts.

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
        TSDataset
            TSDataset with predictions.
        """
        raise NotImplementedError("Method predict isn't currently implemented!")

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
