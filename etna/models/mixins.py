import zipfile
from abc import ABC
from abc import abstractmethod
from copy import deepcopy
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Sequence

import dill
import numpy as np
import pandas as pd
from typing_extensions import Self

from etna.core.mixins import SaveMixin
from etna.datasets.tsdataset import TSDataset
from etna.models.decorators import log_decorator


class ModelForecastingMixin(ABC):
    """Base class for model mixins."""

    @abstractmethod
    def _forecast(self, **kwargs) -> TSDataset:
        pass

    @abstractmethod
    def _predict(self, **kwargs) -> TSDataset:
        pass

    @abstractmethod
    def _forecast_components(self, **kwargs) -> pd.DataFrame:
        pass

    @abstractmethod
    def _predict_components(self, **kwargs) -> pd.DataFrame:
        pass

    def _add_target_components(
        self, ts: TSDataset, predictions: TSDataset, components_prediction_method: Callable, return_components: bool
    ):
        if return_components:
            target_components_df = components_prediction_method(ts=ts)
            predictions.add_target_components(target_components_df=target_components_df)


class NonPredictionIntervalContextIgnorantModelMixin(ModelForecastingMixin):
    """Mixin for models that don't support prediction intervals and don't need context for prediction."""

    def forecast(self, ts: TSDataset, return_components: bool = False) -> TSDataset:
        """Make predictions.

        Parameters
        ----------
        ts:
            Dataset with features
        return_components:
            If True additionally returns forecast components

        Returns
        -------
        :
            Dataset with predictions
        """
        forecast = self._forecast(ts=ts)
        self._add_target_components(
            ts=ts,
            predictions=forecast,
            components_prediction_method=self._forecast_components,
            return_components=return_components,
        )
        return forecast

    def predict(self, ts: TSDataset, return_components: bool = False) -> TSDataset:
        """Make predictions with using true values as autoregression context if possible (teacher forcing).

        Parameters
        ----------
        ts:
            Dataset with features
        return_components:
            If True additionally returns prediction components

        Returns
        -------
        :
            Dataset with predictions
        """
        prediction = self._predict(ts=ts)
        self._add_target_components(
            ts=ts,
            predictions=prediction,
            components_prediction_method=self._predict_components,
            return_components=return_components,
        )
        return prediction


class NonPredictionIntervalContextRequiredModelMixin(ModelForecastingMixin):
    """Mixin for models that don't support prediction intervals and need context for prediction."""

    def forecast(self, ts: TSDataset, prediction_size: int, return_components: bool = False) -> TSDataset:
        """Make predictions.

        Parameters
        ----------
        ts:
            Dataset with features
        prediction_size:
            Number of last timestamps to leave after making prediction.
            Previous timestamps will be used as a context for models that require it.
        return_components:
            If True additionally returns forecast components

        Returns
        -------
        :
            Dataset with predictions
        """
        forecast = self._forecast(ts=ts, prediction_size=prediction_size)
        self._add_target_components(
            ts=ts,
            predictions=forecast,
            components_prediction_method=self._forecast_components,
            return_components=return_components,
        )
        return forecast

    def predict(self, ts: TSDataset, prediction_size: int, return_components: bool = False) -> TSDataset:
        """Make predictions with using true values as autoregression context if possible (teacher forcing).

        Parameters
        ----------
        ts:
            Dataset with features
        prediction_size:
            Number of last timestamps to leave after making prediction.
            Previous timestamps will be used as a context for models that require it.
        return_components:
            If True additionally returns prediction components

        Returns
        -------
        :
            Dataset with predictions
        """
        prediction = self._predict(ts=ts, prediction_size=prediction_size)
        self._add_target_components(
            ts=ts,
            predictions=prediction,
            components_prediction_method=self._predict_components,
            return_components=return_components,
        )
        return prediction


class PredictionIntervalContextIgnorantModelMixin(ModelForecastingMixin):
    """Mixin for models that support prediction intervals and don't need context for prediction."""

    def forecast(
        self,
        ts: TSDataset,
        prediction_interval: bool = False,
        quantiles: Sequence[float] = (0.025, 0.975),
        return_components: bool = False,
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
        return_components:
            If True additionally returns forecast components

        Returns
        -------
        :
            Dataset with predictions
        """
        forecast = self._forecast(ts=ts, prediction_interval=prediction_interval, quantiles=quantiles)
        self._add_target_components(
            ts=ts,
            predictions=forecast,
            components_prediction_method=self._forecast_components,
            return_components=return_components,
        )
        return forecast

    def predict(
        self,
        ts: TSDataset,
        prediction_interval: bool = False,
        quantiles: Sequence[float] = (0.025, 0.975),
        return_components: bool = False,
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
        return_components:
            If True additionally returns prediction components

        Returns
        -------
        :
            Dataset with predictions
        """
        prediction = self._predict(ts=ts, prediction_interval=prediction_interval, quantiles=quantiles)
        self._add_target_components(
            ts=ts,
            predictions=prediction,
            components_prediction_method=self._predict_components,
            return_components=return_components,
        )
        return prediction


class PredictionIntervalContextRequiredModelMixin(ModelForecastingMixin):
    """Mixin for models that support prediction intervals and need context for prediction."""

    def forecast(
        self,
        ts: TSDataset,
        prediction_size: int,
        prediction_interval: bool = False,
        quantiles: Sequence[float] = (0.025, 0.975),
        return_components: bool = False,
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
        return_components:
            If True additionally returns forecast components

        Returns
        -------
        :
            Dataset with predictions
        """
        forecast = self._forecast(
            ts=ts, prediction_size=prediction_size, prediction_interval=prediction_interval, quantiles=quantiles
        )
        self._add_target_components(
            ts=ts,
            predictions=forecast,
            components_prediction_method=self._forecast_components,
            return_components=return_components,
        )
        return forecast

    def predict(
        self,
        ts: TSDataset,
        prediction_size: int,
        prediction_interval: bool = False,
        quantiles: Sequence[float] = (0.025, 0.975),
        return_components: bool = False,
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
        return_components:
            If True additionally returns prediction components

        Returns
        -------
        :
            Dataset with predictions
        """
        prediction = self._predict(
            ts=ts, prediction_size=prediction_size, prediction_interval=prediction_interval, quantiles=quantiles
        )
        self._add_target_components(
            ts=ts,
            predictions=prediction,
            components_prediction_method=self._predict_components,
            return_components=return_components,
        )
        return prediction


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
        model: Any, segment: str, df: pd.DataFrame, prediction_method: Callable, **kwargs
    ) -> pd.DataFrame:
        """Make predictions for one segment."""
        segment_features = df[segment]
        segment_features = segment_features.reset_index()
        dates = segment_features["timestamp"]
        dates.reset_index(drop=True, inplace=True)
        segment_predict = prediction_method(self=model, df=segment_features, **kwargs)
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
        df = ts.to_pandas()
        models = self._get_model()
        for segment in ts.segments:
            if segment not in models:
                raise NotImplementedError("Per-segment models can't make predictions on new segments!")
            segment_model = models[segment]
            segment_predict = self._make_predictions_segment(
                model=segment_model, segment=segment, df=df, prediction_method=prediction_method, **kwargs
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

        prediction_size = kwargs.get("prediction_size")
        if prediction_size is not None:
            ts.df = ts.df.iloc[-prediction_size:]
        return ts

    def _make_component_predictions(self, ts: TSDataset, prediction_method: Callable, **kwargs) -> pd.DataFrame:
        """Make target component predictions.

        Parameters
        ----------
        ts:
            Dataset with features
        prediction_method:
            Method for making components predictions

        Returns
        -------
        :
            DataFrame with predicted components
        """
        features_df = ts.to_pandas()
        result_list = list()
        for segment, model in self._get_model().items():
            segment_predict = self._make_predictions_segment(
                model=model, segment=segment, df=features_df, prediction_method=prediction_method, **kwargs
            )
            result_list.append(segment_predict)

        target_components_df = pd.concat(result_list, ignore_index=True)
        target_components_df = TSDataset.to_dataset(target_components_df)
        return target_components_df

    @log_decorator
    def _forecast(self, ts: TSDataset, **kwargs) -> TSDataset:
        if hasattr(self._base_model, "forecast"):
            return self._make_predictions(ts=ts, prediction_method=self._base_model.__class__.forecast, **kwargs)
        return self._make_predictions(ts=ts, prediction_method=self._base_model.__class__.predict, **kwargs)

    @log_decorator
    def _predict(self, ts: TSDataset, **kwargs) -> TSDataset:
        return self._make_predictions(ts=ts, prediction_method=self._base_model.__class__.predict, **kwargs)

    @log_decorator
    def _forecast_components(self, ts: TSDataset, **kwargs) -> pd.DataFrame:
        if hasattr(self._base_model, "forecast_components"):
            return self._make_component_predictions(
                ts=ts, prediction_method=self._base_model.__class__.forecast_components, **kwargs
            )
        return self._make_component_predictions(
            ts=ts, prediction_method=self._base_model.__class__.predict_components, **kwargs
        )

    @log_decorator
    def _predict_components(self, ts: TSDataset, **kwargs) -> pd.DataFrame:
        return self._make_component_predictions(
            ts=ts, prediction_method=self._base_model.__class__.predict_components, **kwargs
        )


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
        y = prediction_method(self=self._base_model, df=x, **kwargs).reshape(-1, horizon).T
        ts.loc[:, pd.IndexSlice[:, "target"]] = y
        return ts

    def _make_component_predictions(self, ts: TSDataset, prediction_method: Callable, **kwargs) -> pd.DataFrame:
        """Make target component predictions.

        Parameters
        ----------
        ts:
            Dataset with features
        prediction_method:
            Method for making components predictions

        Returns
        -------
        :
            DataFrame with predicted components
        """
        features_df = ts.to_pandas(flatten=True)
        segment_column = features_df["segment"].values
        features_df = features_df.drop(["segment"], axis=1)
        # TODO: make it work with prediction intervals and context
        target_components_df = prediction_method(self=self._base_model, df=features_df, **kwargs)
        target_components_df["segment"] = segment_column
        target_components_df["timestamp"] = features_df["timestamp"]
        target_components_df = TSDataset.to_dataset(target_components_df)
        return target_components_df

    @log_decorator
    def _forecast(self, ts: TSDataset, **kwargs) -> TSDataset:
        if hasattr(self._base_model, "forecast"):
            return self._make_predictions(ts=ts, prediction_method=self._base_model.__class__.forecast, **kwargs)
        return self._make_predictions(ts=ts, prediction_method=self._base_model.__class__.predict, **kwargs)

    @log_decorator
    def _predict(self, ts: TSDataset, **kwargs) -> TSDataset:
        return self._make_predictions(ts=ts, prediction_method=self._base_model.__class__.predict, **kwargs)

    @log_decorator
    def _forecast_components(self, ts: TSDataset, **kwargs) -> pd.DataFrame:
        if hasattr(self._base_model, "forecast_components"):
            return self._make_component_predictions(
                ts=ts, prediction_method=self._base_model.__class__.forecast_components, **kwargs
            )
        return self._make_component_predictions(
            ts=ts, prediction_method=self._base_model.__class__.predict_components, **kwargs
        )

    @log_decorator
    def _predict_components(self, ts: TSDataset, **kwargs) -> pd.DataFrame:
        return self._make_component_predictions(
            ts=ts, prediction_method=self._base_model.__class__.predict_components, **kwargs
        )

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


class SaveNNMixin(SaveMixin):
    """Implementation of ``AbstractSaveable`` torch related classes.

    It saves object to the zip archive with 2 files:

    * metadata.json: contains library version and class name.

    * object.pt: object saved by ``torch.save``.
    """

    def _save_state(self, archive: zipfile.ZipFile):
        import torch

        with archive.open("object.pt", "w", force_zip64=True) as output_file:
            torch.save(self, output_file, pickle_module=dill)

    @classmethod
    def _load_state(cls, archive: zipfile.ZipFile) -> Self:
        import torch

        with archive.open("object.pt", "r") as input_file:
            return torch.load(input_file, pickle_module=dill)
