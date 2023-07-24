from collections.abc import Callable
from copy import deepcopy
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler

from etna import SETTINGS
from etna.core import BaseMixin
from etna.datasets.tsdataset import TSDataset
from etna.loggers import tslogger
from etna.models.base import log_decorator
from etna.models.utils import determine_num_steps

if SETTINGS.torch_required:
    import pytorch_lightning as pl
    from pytorch_forecasting.data import TimeSeriesDataSet
    from pytorch_forecasting.data.encoders import EncoderNormalizer
    from pytorch_forecasting.data.encoders import NaNLabelEncoder
    from pytorch_forecasting.data.encoders import TorchNormalizer
    from torch.utils.data import DataLoader

else:
    TimeSeriesDataSet = None  # type: ignore
    EncoderNormalizer = None  # type: ignore
    NaNLabelEncoder = None  # type: ignore
    TorchNormalizer = None  # type: ignore

NORMALIZER = Union[TorchNormalizer, NaNLabelEncoder, EncoderNormalizer]


class _DeepCopyMixin:
    """Mixin for ``__deepcopy__`` behaviour overriding."""

    def __deepcopy__(self, memo):
        """Drop ``model`` and ``trainer`` attributes while deepcopy."""
        cls = self.__class__
        obj = cls.__new__(cls)
        memo[id(self)] = obj
        for k, v in self.__dict__.items():
            if k in ["model", "trainer"]:
                v = dict()
            setattr(obj, k, deepcopy(v, memo))
            pass
        return obj


class PytorchForecastingDatasetBuilder(BaseMixin):
    """Builder for PytorchForecasting dataset."""

    def __init__(
        self,
        max_encoder_length: int = 30,
        min_encoder_length: Optional[int] = None,
        min_prediction_idx: Optional[int] = None,
        min_prediction_length: Optional[int] = None,
        max_prediction_length: int = 1,
        static_categoricals: Optional[List[str]] = None,
        static_reals: Optional[List[str]] = None,
        time_varying_known_categoricals: Optional[List[str]] = None,
        time_varying_known_reals: Optional[List[str]] = None,
        time_varying_unknown_categoricals: Optional[List[str]] = None,
        time_varying_unknown_reals: Optional[List[str]] = None,
        variable_groups: Optional[Dict[str, List[int]]] = None,
        constant_fill_strategy: Optional[Dict[str, Union[str, float, int, bool]]] = None,
        allow_missing_timesteps: bool = True,
        lags: Optional[Dict[str, List[int]]] = None,
        add_relative_time_idx: bool = True,
        add_target_scales: bool = True,
        add_encoder_length: Union[bool, str] = True,
        target_normalizer: Union[NORMALIZER, str, List[NORMALIZER], Tuple[NORMALIZER]] = "auto",
        categorical_encoders: Optional[Dict[str, NaNLabelEncoder]] = None,
        scalers: Optional[Dict[str, Union[StandardScaler, RobustScaler, TorchNormalizer, EncoderNormalizer]]] = None,
    ):
        """Init dataset builder.

        Parameters here is used for initialization of :py:class:`pytorch_forecasting.data.timeseries.TimeSeriesDataSet` object.
        """
        self.max_encoder_length = max_encoder_length
        self.min_encoder_length = min_encoder_length
        self.min_prediction_idx = min_prediction_idx
        self.min_prediction_length = min_prediction_length
        self.max_prediction_length = max_prediction_length
        self.static_categoricals = static_categoricals if static_categoricals else []
        self.static_reals = static_reals if static_reals else []
        self.time_varying_known_categoricals = (
            time_varying_known_categoricals if time_varying_known_categoricals else []
        )
        self.time_varying_known_reals = time_varying_known_reals if time_varying_known_reals else []
        self.time_varying_unknown_categoricals = (
            time_varying_unknown_categoricals if time_varying_unknown_categoricals else []
        )
        self.time_varying_unknown_reals = time_varying_unknown_reals if time_varying_unknown_reals else []
        self.variable_groups = variable_groups if variable_groups else {}
        self.add_relative_time_idx = add_relative_time_idx
        self.add_target_scales = add_target_scales
        self.add_encoder_length = add_encoder_length
        self.allow_missing_timesteps = allow_missing_timesteps
        self.target_normalizer = target_normalizer
        self.categorical_encoders = categorical_encoders if categorical_encoders else {}
        self.constant_fill_strategy = constant_fill_strategy if constant_fill_strategy else []
        self.lags = lags if lags else {}
        self.scalers = scalers if scalers else {}
        self.pf_dataset_params: Optional[Dict[str, Any]] = None

    def _time_encoder(self, values: List[int]) -> Dict[int, int]:
        encoded_unix_times = dict()
        for idx, value in enumerate(sorted(values)):
            encoded_unix_times[value] = idx
        return encoded_unix_times

    def create_train_dataset(self, ts: TSDataset) -> TimeSeriesDataSet:
        """Create train dataset.

        Parameters
        ----------
        ts:
            Time series dataset.
        """
        df_flat = ts.to_pandas(flatten=True)
        df_flat = df_flat.dropna()

        mapping_time_idx = {x: i for i, x in enumerate(ts.index)}
        df_flat["time_idx"] = df_flat["timestamp"].map(mapping_time_idx)

        self.min_timestamp = df_flat["timestamp"].min()

        if self.time_varying_known_categoricals:
            for feature_name in self.time_varying_known_categoricals:
                df_flat[feature_name] = df_flat[feature_name].astype(str)

        pf_dataset = TimeSeriesDataSet(
            df_flat,
            time_idx="time_idx",
            target="target",
            group_ids=["segment"],
            time_varying_known_reals=self.time_varying_known_reals,
            time_varying_known_categoricals=self.time_varying_known_categoricals,
            time_varying_unknown_reals=self.time_varying_unknown_reals,
            max_encoder_length=self.max_encoder_length,
            max_prediction_length=self.max_prediction_length,
            min_encoder_length=self.min_encoder_length,
            min_prediction_length=self.min_prediction_length,
            add_relative_time_idx=self.add_relative_time_idx,
            add_target_scales=self.add_target_scales,
            add_encoder_length=self.add_encoder_length,
            allow_missing_timesteps=self.allow_missing_timesteps,
            target_normalizer=self.target_normalizer,
            static_categoricals=self.static_categoricals,
            min_prediction_idx=self.min_prediction_idx,
            variable_groups=self.variable_groups,
            constant_fill_strategy=self.constant_fill_strategy,
            lags=self.lags,
            categorical_encoders=self.categorical_encoders,
            scalers=self.scalers,
        )

        self.pf_dataset_params = pf_dataset.get_parameters()

        return pf_dataset

    def create_inference_dataset(self, ts: TSDataset, horizon: int) -> TimeSeriesDataSet:
        """Create inference dataset.

        This method should be used only after ``create_train_dataset`` that is used during model training.

        Parameters
        ----------
        ts:
            Time series dataset.
        horizon:
            Size of prediction to make.

        Raises
        ------
        ValueError:
            if method was used before ``create_train_dataset``
        """
        if self.pf_dataset_params is None:
            raise ValueError(
                "This method should only be called after create_train_dataset. Try to train the model that uses this builder."
            )

        df_flat = ts.to_pandas(flatten=True)
        df_flat = df_flat[df_flat.timestamp >= self.min_timestamp]
        df_flat["target"] = df_flat["target"].fillna(0)

        inference_min_timestamp = df_flat["timestamp"].min()
        time_idx_shift = determine_num_steps(
            start_timestamp=self.min_timestamp, end_timestamp=inference_min_timestamp, freq=ts.freq
        )
        mapping_time_idx = {x: i + time_idx_shift for i, x in enumerate(ts.index)}
        df_flat["time_idx"] = df_flat["timestamp"].map(mapping_time_idx)

        if self.time_varying_known_categoricals:
            for feature_name in self.time_varying_known_categoricals:
                df_flat[feature_name] = df_flat[feature_name].astype(str)

        # `TimeSeriesDataSet.from_parameters` in predict mode ignores `min_prediction_length`,
        # and we can change prediction size only by changing `max_prediction_length`
        dataset_params = deepcopy(self.pf_dataset_params)
        dataset_params["max_prediction_length"] = horizon

        pf_inference_dataset = TimeSeriesDataSet.from_parameters(
            dataset_params, df_flat, predict=True, stop_randomization=True
        )
        return pf_inference_dataset


class PytorchForecastingMixin:
    """Mixin for Pytorch Forecasting models."""

    trainer_params: Dict[str, Any]
    dataset_builder: PytorchForecastingDatasetBuilder
    _from_dataset: Callable
    train_batch_size: int
    test_batch_size: int
    encoder_length: int

    @log_decorator
    def fit(self, ts: TSDataset):
        """
        Fit model.

        Parameters
        ----------
        ts:
            TSDataset to fit.

        Returns
        -------
        :
            model
        """
        self._last_train_timestamp = ts.df.index[-1]
        self._freq = ts.freq

        trainer_params = dict()
        if "logger" not in self.trainer_params:
            self.trainer_params["logger"] = tslogger.pl_loggers
        else:
            self.trainer_params["logger"] += tslogger.pl_loggers

        trainer_params.update(self.trainer_params)

        self.trainer = pl.Trainer(**trainer_params)
        pf_dataset_train = self.dataset_builder.create_train_dataset(ts)
        train_dataloader = pf_dataset_train.to_dataloader(train=True, batch_size=self.train_batch_size)
        self.model = self._from_dataset(pf_dataset_train)
        if self.trainer is not None and self.model is not None:
            self.trainer.fit(self.model, train_dataloader)
        else:
            raise ValueError("Trainer or model is None")
        return self

    def _get_first_prediction_timestamp(self, ts: TSDataset, horizon: int) -> pd.Timestamp:
        return ts.index[-horizon]

    def _is_in_sample_prediction(self, ts: TSDataset, horizon: int) -> bool:
        first_prediction_timestamp = self._get_first_prediction_timestamp(ts=ts, horizon=horizon)
        return first_prediction_timestamp <= self._last_train_timestamp

    def _is_prediction_with_gap(self, ts: TSDataset, horizon: int) -> bool:
        first_prediction_timestamp = self._get_first_prediction_timestamp(ts=ts, horizon=horizon)
        first_timestamp_after_train = pd.date_range(self._last_train_timestamp, periods=2, freq=self._freq)[-1]
        return first_prediction_timestamp > first_timestamp_after_train

    def _make_target_prediction(self, ts: TSDataset, horizon: int) -> Tuple[TSDataset, DataLoader]:
        if self._is_in_sample_prediction(ts=ts, horizon=horizon):
            raise NotImplementedError(
                "This model can't make forecast on history data! "
                "In-sample forecast isn't supported by current implementation."
            )
        elif self._is_prediction_with_gap(ts=ts, horizon=horizon):
            first_prediction_timestamp = self._get_first_prediction_timestamp(ts=ts, horizon=horizon)
            raise NotImplementedError(
                "This model can't make forecast on out-of-sample data that goes after training data with a gap! "
                "You can only forecast from the next point after the last one in the training dataset: "
                f"last train timestamp: {self._last_train_timestamp}, first prediction timestamp is {first_prediction_timestamp}"
            )
        else:
            pass

        if len(ts.df) != horizon + self.encoder_length:
            raise ValueError("Length of dataset must be equal to horizon + max_encoder_length")

        pf_dataset_inference = self.dataset_builder.create_inference_dataset(ts, horizon)

        prediction_dataloader: DataLoader = pf_dataset_inference.to_dataloader(
            train=False, batch_size=self.test_batch_size
        )

        # shape (segments, encoder_length)
        predicts = self.model.predict(prediction_dataloader).numpy()

        ts.df = ts.df.iloc[-horizon:]
        ts.loc[:, pd.IndexSlice[:, "target"]] = predicts.T[:horizon]
        return ts, prediction_dataloader
