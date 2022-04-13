import inspect
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler

from etna import SETTINGS
from etna.datasets.tsdataset import TSDataset
from etna.transforms.base import Transform

if SETTINGS.torch_required:
    from pytorch_forecasting.data import TimeSeriesDataSet
    from pytorch_forecasting.data.encoders import EncoderNormalizer
    from pytorch_forecasting.data.encoders import NaNLabelEncoder
    from pytorch_forecasting.data.encoders import TorchNormalizer
else:
    TimeSeriesDataSet = None  # type: ignore
    EncoderNormalizer = None  # type: ignore
    NaNLabelEncoder = None  # type: ignore
    TorchNormalizer = None  # type: ignore

NORMALIZER = Union[TorchNormalizer, NaNLabelEncoder, EncoderNormalizer]


class PytorchForecastingTransform(Transform):
    """Transform for models from PytorchForecasting library.

    Notes
    -----
    This transform should be added at the very end of ``transforms`` parameter.
    """

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
        """Init transform.

        Parameters here is used for initialization of :py:class:`pytorch_forecasting.data.timeseries.TimeSeriesDataSet` object.
        """
        super().__init__()
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
        self.pf_dataset_predict: Optional[TimeSeriesDataSet] = None

    def fit(self, df: pd.DataFrame) -> "PytorchForecastingTransform":
        """
        Fit TimeSeriesDataSet.

        Parameters
        ----------
        df:
            data to be fitted.

        Returns
        -------
            PytorchForecastingTransform
        """
        self.freq = pd.infer_freq(df.index)
        ts = TSDataset(df, self.freq)
        df_flat = ts.to_pandas(flatten=True)
        df_flat = df_flat.dropna()
        self.min_timestamp = df_flat.timestamp.min()

        if self.time_varying_known_categoricals:
            for feature_name in self.time_varying_known_categoricals:
                df_flat[feature_name] = df_flat[feature_name].astype(str)

        # making time_idx feature.
        # it's needed for pytorch-forecasting for proper train-test split.
        # it should be incremented by 1 for every new timestamp.
        df_flat["time_idx"] = (df_flat["timestamp"] - self.min_timestamp) // pd.Timedelta("1s")
        encoded_unix_times = self._time_encoder(list(df_flat.time_idx.unique()))
        df_flat["time_idx"] = df_flat["time_idx"].apply(lambda x: encoded_unix_times[x])

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

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform raw df to TimeSeriesDataSet.

        Parameters
        ----------
        df:
            data to be transformed.

        Returns
        -------
            DataFrame

        Notes
        -----
        We save TimeSeriesDataSet in instance to use it in the model.
        It`s not right pattern of using Transforms and TSDataset.
        """
        ts = TSDataset(df, self.freq)
        df_flat = ts.to_pandas(flatten=True)
        df_flat = df_flat[df_flat.timestamp >= self.min_timestamp]
        df_flat["target"] = df_flat["target"].fillna(0)

        df_flat["time_idx"] = (df_flat["timestamp"] - self.min_timestamp) // pd.Timedelta("1s")
        encoded_unix_times = self._time_encoder(list(df_flat.time_idx.unique()))
        df_flat["time_idx"] = df_flat["time_idx"].apply(lambda x: encoded_unix_times[x])

        if self.time_varying_known_categoricals:
            for feature_name in self.time_varying_known_categoricals:
                df_flat[feature_name] = df_flat[feature_name].astype(str)

        if inspect.stack()[1].function == "make_future":
            pf_dataset_predict = TimeSeriesDataSet.from_parameters(
                self.pf_dataset_params, df_flat, predict=True, stop_randomization=True
            )
            self.pf_dataset_predict = pf_dataset_predict
        else:
            pf_dataset_train = TimeSeriesDataSet.from_parameters(self.pf_dataset_params, df_flat)
            self.pf_dataset_train = pf_dataset_train
        return df

    def _time_encoder(self, values: List[int]) -> Dict[int, int]:
        encoded_unix_times = dict()
        for idx, value in enumerate(sorted(values)):
            encoded_unix_times[value] = idx
        return encoded_unix_times
