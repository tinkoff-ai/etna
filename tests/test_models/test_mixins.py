import json
import pathlib
from unittest.mock import MagicMock
from unittest.mock import patch
from zipfile import ZipFile

import dill
import numpy as np
import pytest

from etna import SETTINGS
from etna.datasets import TSDataset

if SETTINGS.torch_required:
    import torch

import pandas as pd

from etna.models.base import BaseAdapter
from etna.models.mixins import MultiSegmentModelMixin
from etna.models.mixins import NonPredictionIntervalContextIgnorantModelMixin
from etna.models.mixins import NonPredictionIntervalContextRequiredModelMixin
from etna.models.mixins import PerSegmentModelMixin
from etna.models.mixins import PredictionIntervalContextIgnorantModelMixin
from etna.models.mixins import PredictionIntervalContextRequiredModelMixin
from etna.models.mixins import SaveNNMixin


class DummyPredictAdapter(BaseAdapter):
    def fit(self, df: pd.DataFrame, **kwargs) -> "DummyPredictAdapter":
        return self

    def predict(self, df: pd.DataFrame, **kwargs) -> np.ndarray:
        df["target"] = 200
        return df["target"].values

    def predict_components(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df["target_component_a"] = 20
        df["target_component_b"] = 180
        df = df.drop(columns=["target"])
        return df

    def get_model(self) -> "DummyPredictAdapter":
        return self


class DummyForecastPredictAdapter(DummyPredictAdapter):
    def fit(self, df: pd.DataFrame, **kwargs) -> "DummyForecastPredictAdapter":
        return self

    def forecast(self, df: pd.DataFrame, **kwargs) -> np.ndarray:
        df["target"] = 100
        return df["target"].values

    def forecast_components(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df["target_component_a"] = 10
        df["target_component_b"] = 90
        df = df.drop(columns=["target"])
        return df


class DummyModelBase:
    def _forecast(self, ts: TSDataset, **kwargs) -> TSDataset:
        ts.loc[pd.IndexSlice[:], pd.IndexSlice[:, "target"]] = 100
        return ts

    def _predict(self, ts: TSDataset, **kwargs) -> TSDataset:
        ts.loc[pd.IndexSlice[:], pd.IndexSlice[:, "target"]] = 200
        return ts

    def _forecast_components(self, ts: TSDataset, **kwargs) -> pd.DataFrame:
        df = ts.to_pandas(flatten=True, features=["target"])
        df["target_component_a"] = 10
        df["target_component_b"] = 90
        df = df.drop(columns=["target"])
        df = TSDataset.to_dataset(df)
        return df

    def _predict_components(self, ts: TSDataset, **kwargs) -> pd.DataFrame:
        df = ts.to_pandas(flatten=True, features=["target"])
        df["target_component_a"] = 20
        df["target_component_b"] = 180
        df = df.drop(columns=["target"])
        df = TSDataset.to_dataset(df)
        return df


class NonPredictionIntervalContextIgnorantDummyModel(DummyModelBase, NonPredictionIntervalContextIgnorantModelMixin):
    pass


class NonPredictionIntervalContextRequiredDummyModel(DummyModelBase, NonPredictionIntervalContextRequiredModelMixin):
    pass


class PredictionIntervalContextIgnorantDummyModel(DummyModelBase, PredictionIntervalContextIgnorantModelMixin):
    pass


class PredictionIntervalContextRequiredDummyModel(DummyModelBase, PredictionIntervalContextRequiredModelMixin):
    pass


@pytest.fixture()
def regression_base_model_mock():
    cls = MagicMock()
    del cls.forecast

    model = MagicMock()
    model.__class__ = cls
    del model.forecast
    return model


@pytest.fixture()
def autoregression_base_model_mock():
    cls = MagicMock()

    model = MagicMock()
    model.__class__ = cls
    return model


@pytest.mark.parametrize("mixin_constructor", [PerSegmentModelMixin, MultiSegmentModelMixin])
@pytest.mark.parametrize(
    "base_model_name, called_method_name, expected_method_name",
    [
        ("regression_base_model_mock", "_forecast", "predict"),
        ("autoregression_base_model_mock", "_forecast", "forecast"),
        ("regression_base_model_mock", "_predict", "predict"),
        ("autoregression_base_model_mock", "_predict", "predict"),
    ],
)
def test_calling_private_prediction(
    base_model_name, called_method_name, expected_method_name, mixin_constructor, request
):
    base_model = request.getfixturevalue(base_model_name)
    ts = MagicMock()
    mixin = mixin_constructor(base_model=base_model)
    mixin._make_predictions = MagicMock()
    to_call = getattr(mixin, called_method_name)
    to_call(ts=ts)
    mixin._make_predictions.assert_called_once_with(
        ts=ts, prediction_method=getattr(base_model.__class__, expected_method_name)
    )


class DummyNN(SaveNNMixin):
    def __init__(self, a, b):
        self.a = torch.tensor(a)
        self.b = torch.tensor(b)


def test_save_nn_mixin_save(tmp_path):
    dummy = DummyNN(a=1, b=2)
    dir_path = pathlib.Path(tmp_path)
    path = dir_path.joinpath("dummy.zip")

    dummy.save(path)

    with ZipFile(path, "r") as zip_file:
        files = zip_file.namelist()
        assert sorted(files) == ["metadata.json", "object.pt"]

        with zip_file.open("metadata.json", "r") as input_file:
            metadata_bytes = input_file.read()
        metadata_str = metadata_bytes.decode("utf-8")
        metadata = json.loads(metadata_str)
        assert sorted(metadata.keys()) == ["class", "etna_version"]
        assert metadata["class"] == "tests.test_models.test_mixins.DummyNN"

        with zip_file.open("object.pt", "r") as input_file:
            loaded_dummy = torch.load(input_file, pickle_module=dill)
        assert loaded_dummy.a == dummy.a
        assert loaded_dummy.b == dummy.b


def test_save_mixin_load_ok(recwarn, tmp_path):
    dummy = DummyNN(a=1, b=2)
    dir_path = pathlib.Path(tmp_path)
    path = dir_path.joinpath("dummy.zip")

    dummy.save(path)
    loaded_dummy = DummyNN.load(path)

    assert loaded_dummy.a == dummy.a
    assert loaded_dummy.b == dummy.b
    assert len(recwarn) == 0


@pytest.mark.parametrize(
    "save_version, load_version", [((1, 5, 0), (2, 5, 0)), ((2, 5, 0), (1, 5, 0)), ((1, 5, 0), (1, 3, 0))]
)
@patch("etna.core.mixins.get_etna_version")
def test_save_mixin_load_warning(get_version_mock, save_version, load_version, tmp_path):
    dummy = DummyNN(a=1, b=2)
    dir_path = pathlib.Path(tmp_path)
    path = dir_path.joinpath("dummy.zip")

    get_version_mock.return_value = save_version
    dummy.save(path)

    save_version_str = ".".join([str(x) for x in save_version])
    load_version_str = ".".join([str(x) for x in load_version])
    with pytest.warns(
        UserWarning,
        match=f"The object was saved under etna version {save_version_str} but running version is {load_version_str}",
    ):
        get_version_mock.return_value = load_version
        _ = DummyNN.load(path)


@pytest.mark.parametrize(
    "mixin_constructor, call_params",
    [
        (NonPredictionIntervalContextIgnorantDummyModel, {}),
        (NonPredictionIntervalContextRequiredDummyModel, {"prediction_size": 10}),
        (PredictionIntervalContextIgnorantDummyModel, {}),
        (PredictionIntervalContextRequiredDummyModel, {"prediction_size": 10}),
    ],
)
@pytest.mark.parametrize("method_name, expected_target", [("forecast", 100), ("predict", 200)])
def test_model_mixins_predict_without_target_components(
    example_tsds,
    mixin_constructor,
    call_params,
    method_name,
    expected_target,
    expected_columns=["timestamp", "segment", "target"],
):
    mixin = mixin_constructor()
    to_call = getattr(mixin, method_name)
    forecast = to_call(ts=example_tsds, return_components=False, **call_params).to_pandas(flatten=True)
    assert sorted(forecast.columns) == sorted(expected_columns)
    assert (forecast["target"] == expected_target).all()


@pytest.mark.parametrize(
    "mixin_constructor, call_params",
    [
        (NonPredictionIntervalContextIgnorantDummyModel, {}),
        (NonPredictionIntervalContextRequiredDummyModel, {"prediction_size": 10}),
        (PredictionIntervalContextIgnorantDummyModel, {}),
        (PredictionIntervalContextRequiredDummyModel, {"prediction_size": 10}),
    ],
)
@pytest.mark.parametrize(
    "method_name, expected_target, expected_component_a, expected_component_b",
    [("forecast", 100, 10, 90), ("predict", 200, 20, 180)],
)
def test_model_mixins_prediction_methods_with_target_components(
    example_tsds,
    mixin_constructor,
    call_params,
    method_name,
    expected_target,
    expected_component_a,
    expected_component_b,
    expected_columns=["timestamp", "segment", "target", "target_component_a", "target_component_b"],
):
    mixin = mixin_constructor()
    to_call = getattr(mixin, method_name)
    forecast = to_call(ts=example_tsds, return_components=True, **call_params).to_pandas(flatten=True)
    assert sorted(forecast.columns) == sorted(expected_columns)
    assert (forecast["target"] == expected_target).all()
    assert (forecast["target_component_a"] == expected_component_a).all()
    assert (forecast["target_component_b"] == expected_component_b).all()


@pytest.mark.parametrize("mixin_constructor", [PerSegmentModelMixin, MultiSegmentModelMixin])
@pytest.mark.parametrize(
    "method_name, adapter_constructor, expected_target",
    [
        ("_forecast", DummyForecastPredictAdapter, 100),
        ("_predict", DummyForecastPredictAdapter, 200),
        ("_forecast", DummyPredictAdapter, 200),
        ("_predict", DummyPredictAdapter, 200),
    ],
)
def test_mixin_implementations_prediction_methods(
    example_tsds,
    mixin_constructor,
    method_name,
    adapter_constructor,
    expected_target,
    expected_columns=["timestamp", "segment", "target"],
):
    mixin = mixin_constructor(base_model=adapter_constructor())
    mixin = mixin.fit(ts=example_tsds)
    to_call = getattr(mixin, method_name)
    forecast = to_call(ts=example_tsds).to_pandas(flatten=True)
    assert sorted(forecast.columns) == sorted(expected_columns)
    assert (forecast["target"] == expected_target).all()


@pytest.mark.parametrize("mixin_constructor", [PerSegmentModelMixin, MultiSegmentModelMixin])
@pytest.mark.parametrize(
    "method_name, adapter_constructor, expected_component_a, expected_component_b",
    [
        ("_forecast_components", DummyForecastPredictAdapter, 10, 90),
        ("_predict_components", DummyForecastPredictAdapter, 20, 180),
        ("_forecast_components", DummyPredictAdapter, 20, 180),
        ("_predict_components", DummyPredictAdapter, 20, 180),
    ],
)
def test_mixin_implementations_prediction_components_methods(
    example_tsds,
    mixin_constructor,
    method_name,
    adapter_constructor,
    expected_component_a,
    expected_component_b,
    expected_columns=["timestamp", "segment", "target_component_a", "target_component_b"],
):
    mixin = mixin_constructor(base_model=adapter_constructor())
    mixin = mixin.fit(ts=example_tsds)
    to_call = getattr(mixin, method_name)
    forecast = TSDataset.to_flatten(to_call(ts=example_tsds))
    assert sorted(forecast.columns) == sorted(expected_columns)
    assert (forecast["target_component_a"] == expected_component_a).all()
    assert (forecast["target_component_b"] == expected_component_b).all()
