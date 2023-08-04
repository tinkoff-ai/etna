import pathlib
import tempfile
import zipfile
from copy import deepcopy
from typing import Dict
from typing import Optional
from typing import Sequence

import numpy as np
import pandas as pd
from typing_extensions import Self
from typing_extensions import get_args

from etna.core import SaveMixin
from etna.core import load
from etna.datasets import TSDataset
from etna.distributions import BaseDistribution
from etna.models import ModelType
from etna.models import NonPredictionIntervalContextIgnorantAbstractModel
from etna.models import NonPredictionIntervalContextRequiredAbstractModel
from etna.models import NonPredictionIntervalModelType
from etna.models import PredictionIntervalContextIgnorantAbstractModel
from etna.models import PredictionIntervalContextRequiredAbstractModel
from etna.transforms import Transform


class ModelPipelinePredictMixin:
    """Mixin for pipelines with model inside with implementation of ``_predict`` method."""

    model: ModelType
    transforms: Sequence[Transform]

    def _create_ts(self, ts: TSDataset, start_timestamp: pd.Timestamp, end_timestamp: pd.Timestamp) -> TSDataset:
        """Create ``TSDataset`` to make predictions on."""
        df = deepcopy(ts.raw_df)
        df_exog = deepcopy(ts.df_exog)
        freq = deepcopy(ts.freq)
        known_future = deepcopy(ts.known_future)

        df_to_transform = df[:end_timestamp]

        cur_ts = TSDataset(
            df=df_to_transform,
            df_exog=df_exog,
            freq=freq,
            known_future=known_future,
            hierarchical_structure=ts.hierarchical_structure,
        )

        cur_ts.transform(transforms=self.transforms)

        # correct start_timestamp taking into account context size
        timestamp_indices = pd.Series(np.arange(len(df.index)), index=df.index)
        start_idx = timestamp_indices[start_timestamp]
        start_idx = max(0, start_idx - self.model.context_size)
        start_timestamp = timestamp_indices.index[start_idx]

        cur_ts.df = cur_ts.df[start_timestamp:end_timestamp]
        return cur_ts

    def _determine_prediction_size(
        self, ts: TSDataset, start_timestamp: pd.Timestamp, end_timestamp: pd.Timestamp
    ) -> int:
        timestamp_indices = pd.Series(np.arange(len(ts.index)), index=ts.index)
        timestamps = timestamp_indices[start_timestamp:end_timestamp]
        return len(timestamps)

    def _predict(
        self,
        ts: TSDataset,
        start_timestamp: pd.Timestamp,
        end_timestamp: pd.Timestamp,
        prediction_interval: bool,
        quantiles: Sequence[float],
        return_components: bool = False,
    ) -> TSDataset:
        predict_ts = self._create_ts(ts=ts, start_timestamp=start_timestamp, end_timestamp=end_timestamp)
        prediction_size = self._determine_prediction_size(
            ts=ts, start_timestamp=start_timestamp, end_timestamp=end_timestamp
        )

        if prediction_interval and isinstance(self.model, get_args(NonPredictionIntervalModelType)):
            raise NotImplementedError(f"Model {self.model.__class__.__name__} doesn't support prediction intervals!")

        if isinstance(self.model, NonPredictionIntervalContextIgnorantAbstractModel):
            results = self.model.predict(ts=predict_ts, return_components=return_components)
        elif isinstance(self.model, NonPredictionIntervalContextRequiredAbstractModel):
            results = self.model.predict(
                ts=predict_ts, prediction_size=prediction_size, return_components=return_components
            )
        elif isinstance(self.model, PredictionIntervalContextIgnorantAbstractModel):
            results = self.model.predict(
                ts=predict_ts,
                prediction_interval=prediction_interval,
                quantiles=quantiles,
                return_components=return_components,
            )
        elif isinstance(self.model, PredictionIntervalContextRequiredAbstractModel):
            results = self.model.predict(
                ts=predict_ts,
                prediction_size=prediction_size,
                prediction_interval=prediction_interval,
                quantiles=quantiles,
                return_components=return_components,
            )
        else:
            raise NotImplementedError(f"Unknown model type: {self.model.__class__.__name__}!")

        results.inverse_transform(self.transforms)
        return results


class ModelPipelineParamsToTuneMixin:
    """Mixin for pipelines with model inside with implementation of ``params_to_tune`` method."""

    model: ModelType
    transforms: Sequence[Transform]

    def params_to_tune(self) -> Dict[str, BaseDistribution]:
        """Get hyperparameter grid to tune.

        Parameters for model has prefix "model.", e.g. "model.alpha".

        Parameters for transforms has prefix "transforms.idx.", e.g. "transforms.0.mode".

        Returns
        -------
        :
            Grid with parameters from model and transforms.
        """
        all_params = {}
        for key, value in self.model.params_to_tune().items():
            new_key = f"model.{key}"
            all_params[new_key] = value

        for i, transform in enumerate(self.transforms):
            for key, value in transform.params_to_tune().items():
                new_key = f"transforms.{i}.{key}"
                all_params[new_key] = value

        return all_params


class SaveModelPipelineMixin(SaveMixin):
    """Implementation of ``AbstractSaveable`` abstract class for pipelines with model inside.

    It saves object to the zip archive with 4 entities:

    * metadata.json: contains library version and class name.

    * object.pkl: pickled without model, transforms and ts.

    * model.zip: saved model.

    * transforms: folder with saved transforms.
    """

    model: ModelType
    transforms: Sequence[Transform]
    ts: Optional[TSDataset]

    def save(self, path: pathlib.Path):
        """Save the object.

        Parameters
        ----------
        path:
            Path to save object to.
        """
        model = self.model
        transforms = self.transforms
        ts = self.ts

        try:
            # extract attributes we can't easily save
            delattr(self, "model")
            delattr(self, "transforms")
            delattr(self, "ts")

            # save the remaining part
            super().save(path=path)
        finally:
            self.model = model
            self.transforms = transforms
            self.ts = ts

        with zipfile.ZipFile(path, "a") as archive:
            with tempfile.TemporaryDirectory() as _temp_dir:
                temp_dir = pathlib.Path(_temp_dir)

                # save model separately
                model_save_path = temp_dir / "model.zip"
                model.save(model_save_path)
                archive.write(model_save_path, "model.zip")

                # save transforms separately
                transforms_dir = temp_dir / "transforms"
                transforms_dir.mkdir()
                num_digits = 8
                for i, transform in enumerate(transforms):
                    save_name = f"{i:0{num_digits}d}.zip"
                    transform_save_path = transforms_dir / save_name
                    transform.save(transform_save_path)
                    archive.write(transform_save_path, f"transforms/{save_name}")

    @classmethod
    def load(cls, path: pathlib.Path, ts: Optional[TSDataset] = None) -> Self:
        """Load an object.

        Warning
        -------
        This method uses :py:mod:`dill` module which is not secure.
        It is possible to construct malicious data which will execute arbitrary code during loading.
        Never load data that could have come from an untrusted source, or that could have been tampered with.

        Parameters
        ----------
        path:
            Path to load object from.
        ts:
            TSDataset to set into loaded pipeline.

        Returns
        -------
        :
            Loaded object.
        """
        obj = super().load(path=path)
        obj.ts = deepcopy(ts)

        with zipfile.ZipFile(path, "r") as archive:
            with tempfile.TemporaryDirectory() as _temp_dir:
                temp_dir = pathlib.Path(_temp_dir)

                archive.extractall(temp_dir)

                # load model
                model_path = temp_dir / "model.zip"
                obj.model = load(model_path)

                # load transforms
                transforms_dir = temp_dir / "transforms"
                transforms = []

                if transforms_dir.exists():
                    for path in sorted(transforms_dir.iterdir()):
                        transforms.append(load(path))

                obj.transforms = transforms

        return obj
