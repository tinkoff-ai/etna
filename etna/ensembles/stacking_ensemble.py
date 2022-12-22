import warnings
from copy import deepcopy
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Union
from typing import cast

import numpy as np
import pandas as pd
from joblib import Parallel
from joblib import delayed
from sklearn.base import RegressorMixin
from sklearn.linear_model import LinearRegression
from typing_extensions import Literal

from etna.datasets import TSDataset
from etna.ensembles.mixins import EnsembleMixin
from etna.ensembles.mixins import SaveEnsembleMixin
from etna.loggers import tslogger
from etna.metrics import MAE
from etna.pipeline.base import BasePipeline


class StackingEnsemble(EnsembleMixin, SaveEnsembleMixin, BasePipeline):
    """StackingEnsemble is a pipeline that forecast future using the metamodel to combine the forecasts of the base models.

    Examples
    --------
    >>> from etna.datasets import generate_ar_df
    >>> from etna.datasets import TSDataset
    >>> from etna.ensembles import VotingEnsemble
    >>> from etna.models import NaiveModel
    >>> from etna.models import MovingAverageModel
    >>> from etna.pipeline import Pipeline
    >>> import pandas as pd
    >>> pd.options.display.float_format = '{:,.2f}'.format
    >>> df = generate_ar_df(periods=100, start_time="2021-06-01", ar_coef=[0.8], n_segments=3)
    >>> df_ts_format = TSDataset.to_dataset(df)
    >>> ts = TSDataset(df_ts_format, "D")
    >>> ma_pipeline = Pipeline(model=MovingAverageModel(window=5), transforms=[], horizon=7)
    >>> naive_pipeline = Pipeline(model=NaiveModel(lag=10), transforms=[], horizon=7)
    >>> ensemble = StackingEnsemble(pipelines=[ma_pipeline, naive_pipeline])
    >>> _ = ensemble.fit(ts=ts)
    >>> forecast = ensemble.forecast()
    >>> forecast[:,:,"target"]
    segment    segment_0 segment_1 segment_2
    feature       target    target    target
    timestamp
    2021-09-09      0.70      1.47      0.20
    2021-09-10      0.62      1.53      0.26
    2021-09-11      0.50      1.78      0.36
    2021-09-12      0.37      1.88      0.21
    2021-09-13      0.46      1.87      0.25
    2021-09-14      0.44      1.49      0.21
    2021-09-15      0.36      1.56      0.30
    """

    def __init__(
        self,
        pipelines: List[BasePipeline],
        final_model: Optional[RegressorMixin] = None,
        n_folds: int = 3,
        features_to_use: Union[None, Literal["all"], List[str]] = None,
        n_jobs: int = 1,
        joblib_params: Optional[Dict[str, Any]] = None,
    ):
        """Init StackingEnsemble.

        Parameters
        ----------
        pipelines:
            List of pipelines that should be used in ensemble.
        final_model:
            Regression model with fit/predict interface which will be used to combine the base estimators.
        n_folds:
            Number of folds to use in the backtest. Backtest is not used for model evaluation but for prediction.
        features_to_use:
            Features except the forecasts of the base models to use in the ``final_model``.
        n_jobs:
            Number of jobs to run in parallel.
        joblib_params:
            Additional parameters for :py:class:`joblib.Parallel`.

        Raises
        ------
        ValueError:
            If the number of the pipelines is less than 2 or pipelines have different horizons.
        """
        self._validate_pipeline_number(pipelines=pipelines)
        self.pipelines = pipelines
        self.final_model = LinearRegression() if final_model is None else final_model
        self._validate_backtest_n_folds(n_folds)
        self.n_folds = n_folds
        self.features_to_use = features_to_use
        self.filtered_features_for_final_model: Union[None, Set[str]] = None
        self.n_jobs = n_jobs
        if joblib_params is None:
            self.joblib_params = dict(verbose=11, backend="multiprocessing", mmap_mode="c")
        else:
            self.joblib_params = joblib_params
        super().__init__(horizon=self._get_horizon(pipelines=pipelines))

    def _filter_features_to_use(self, forecasts: List[TSDataset]) -> Union[None, Set[str]]:
        """Return all the features from ``features_to_use`` which can be obtained from base models' forecasts."""
        features_df = pd.concat([forecast.df for forecast in forecasts], axis=1)
        available_features = set(features_df.columns.get_level_values("feature")) - {"fold_number"}
        features_to_use = self.features_to_use
        if features_to_use is None:
            return None
        elif features_to_use == "all":
            return available_features - {"target"}
        elif isinstance(features_to_use, list):
            features_to_use_unique = set(features_to_use)
            if len(features_to_use_unique) == 0:
                return None
            elif features_to_use_unique.issubset(available_features):
                return features_to_use_unique
            else:
                unavailable_features = features_to_use_unique - available_features
                warnings.warn(f"Features {unavailable_features} are not found and will be dropped!")
                return features_to_use_unique.intersection(available_features)
        else:
            warnings.warn(
                "Feature list is passed in the wrong format."
                "Only the base models' forecasts will be used for the final forecast."
            )
            return None

    def _backtest_pipeline(self, pipeline: BasePipeline, ts: TSDataset) -> TSDataset:
        """Get forecasts from backtest for given pipeline."""
        with tslogger.disable():
            _, forecasts, _ = pipeline.backtest(ts=ts, metrics=[MAE()], n_folds=self.n_folds)
        forecasts = TSDataset(df=forecasts, freq=ts.freq)
        return forecasts

    def fit(self, ts: TSDataset) -> "StackingEnsemble":
        """Fit the ensemble.

        Parameters
        ----------
        ts:
            TSDataset to fit ensemble.

        Returns
        -------
        self:
            Fitted ensemble.
        """
        self.ts = ts

        # Get forecasts from base models on backtest to fit the final model on
        forecasts = Parallel(n_jobs=self.n_jobs, **self.joblib_params)(
            delayed(self._backtest_pipeline)(pipeline=pipeline, ts=deepcopy(ts)) for pipeline in self.pipelines
        )

        # Fit the final model
        self.filtered_features_for_final_model = self._filter_features_to_use(forecasts)
        x, y = self._make_features(forecasts=forecasts, train=True)
        self.final_model.fit(x, y)

        # Fit the base models
        self.pipelines = Parallel(n_jobs=self.n_jobs, **self.joblib_params)(
            delayed(self._fit_pipeline)(pipeline=pipeline, ts=deepcopy(ts)) for pipeline in self.pipelines
        )
        return self

    def _make_features(
        self, forecasts: List[TSDataset], train: bool = False
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Prepare features for the ``final_model``."""
        if self.ts is None:
            raise ValueError("StackingEnsemble is not fitted! Fit the StackingEnsemble before calling forecast method.")

        # Stack targets from the forecasts
        targets = [
            forecast[:, :, "target"].rename({"target": f"regressor_target_{i}"}, axis=1)
            for i, forecast in enumerate(forecasts)
        ]
        targets = pd.concat(targets, axis=1)

        # Get features from filtered_features_for_final_model
        features = pd.DataFrame()
        if self.filtered_features_for_final_model is not None:
            features_in_forecasts = [
                list(
                    set(forecast.columns.get_level_values("feature")).intersection(
                        self.filtered_features_for_final_model
                    )
                )
                for forecast in forecasts
            ]
            features = pd.concat(
                [forecast[:, :, features_in_forecasts[i]] for i, forecast in enumerate(forecasts)], axis=1
            )
            features = features.loc[:, ~features.columns.duplicated()]
        features_df = pd.concat([features, targets], axis=1)

        # Flatten the features to fit the sklearn interface
        x = pd.concat([features_df.loc[:, segment] for segment in self.ts.segments], axis=0)
        if train:
            y = pd.concat(
                [
                    self.ts[forecasts[0].index.min() : forecasts[0].index.max(), segment, "target"]
                    for segment in self.ts.segments
                ],
                axis=0,
            )
            return x, y
        else:
            return x, None

    def _process_forecasts(self, forecasts: List[TSDataset]) -> TSDataset:
        x, _ = self._make_features(forecasts=forecasts, train=False)
        self.ts = cast(TSDataset, self.ts)
        y = self.final_model.predict(x)
        num_segments = len(forecasts[0].segments)
        y = y.reshape(num_segments, -1).T
        num_timestamps = y.shape[0]

        # Format the forecast into TSDataset
        segment_col = [segment for segment in self.ts.segments for _ in range(num_timestamps)]
        x.loc[:, "segment"] = segment_col
        x.loc[:, "timestamp"] = x.index.values
        df_exog = TSDataset.to_dataset(x)

        df = forecasts[0][:, :, "target"].copy()
        df.loc[pd.IndexSlice[:], pd.IndexSlice[:, "target"]] = np.NAN

        result = TSDataset(df=df, freq=self.ts.freq, df_exog=df_exog)
        result.loc[pd.IndexSlice[:], pd.IndexSlice[:, "target"]] = y
        return result

    def _forecast(self) -> TSDataset:
        """Make predictions.

        Compute the combination of pipelines' forecasts using ``final_model``
        """
        if self.ts is None:
            raise ValueError("Something went wrong, ts is None!")
        forecasts = Parallel(n_jobs=self.n_jobs, **self.joblib_params)(
            delayed(self._forecast_pipeline)(pipeline=pipeline) for pipeline in self.pipelines
        )
        forecast = self._process_forecasts(forecasts=forecasts)
        return forecast

    def _predict(
        self,
        ts: TSDataset,
        start_timestamp: pd.Timestamp,
        end_timestamp: pd.Timestamp,
        prediction_interval: bool,
        quantiles: Sequence[float],
    ) -> TSDataset:
        if prediction_interval:
            raise NotImplementedError(f"Ensemble {self.__class__.__name__} doesn't support prediction intervals!")

        predictions = Parallel(n_jobs=self.n_jobs, **self.joblib_params)(
            delayed(self._predict_pipeline)(
                ts=ts, pipeline=pipeline, start_timestamp=start_timestamp, end_timestamp=end_timestamp
            )
            for pipeline in self.pipelines
        )
        prediction = self._process_forecasts(forecasts=predictions)
        return prediction
