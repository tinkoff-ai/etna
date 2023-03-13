import json
import pathlib
from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch
from zipfile import ZipFile

import dill
import pytest

from etna import SETTINGS
from etna.datasets import TSDataset

if SETTINGS.torch_required:
    import torch

import pandas as pd

from etna.models.mixins import MultiSegmentModelMixin
from etna.models.mixins import NonPredictionIntervalContextIgnorantModelMixin
from etna.models.mixins import NonPredictionIntervalContextRequiredModelMixin
from etna.models.mixins import PerSegmentModelMixin
from etna.models.mixins import PredictionIntervalContextIgnorantModelMixin
from etna.models.mixins import PredictionIntervalContextRequiredModelMixin
from etna.models.mixins import SaveNNMixin


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


@pytest.fixture
def target_components_df():
    timestamp = pd.date_range("2021-01-01", "2021-01-15")
    df = pd.DataFrame({"timestamp": timestamp, "target_component_a": 1, "target_component_b": 2, "segment": 1})
    df = TSDataset.to_dataset(df)
    return df


@pytest.fixture
def ts_without_target_components():
    timestamp = pd.date_range("2021-01-01", "2021-01-15")
    df = pd.DataFrame({"timestamp": timestamp, "target": 3, "segment": 1})
    df = TSDataset.to_dataset(df)
    ts = TSDataset(df=df, freq="D")
    return ts


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
        (PredictionIntervalContextIgnorantModelMixin, {}),
        (NonPredictionIntervalContextIgnorantModelMixin, {}),
        (PredictionIntervalContextRequiredModelMixin, {"prediction_size": 10}),
        (NonPredictionIntervalContextRequiredModelMixin, {"prediction_size": 10}),
    ],
)
@pytest.mark.parametrize("return_components", (True, False))
def test_model_mixins_calls_add_target_components_in_forecast(mixin_constructor, return_components, call_params):
    with patch.multiple(mixin_constructor, __abstractmethods__=set()):
        ts = Mock()
        forecast_ts = Mock(spec=TSDataset)
        mixin = mixin_constructor()
        mixin._forecast = Mock(return_value=forecast_ts)
        mixin._add_target_components = Mock()

        _ = mixin.forecast(ts=ts, return_components=return_components, **call_params)

        mixin._add_target_components.assert_called_with(
            ts=ts,
            predictions=forecast_ts,
            components_prediction_method=mixin._forecast_components,
            return_components=return_components,
        )


@pytest.mark.parametrize(
    "mixin_constructor, call_params",
    [
        (PredictionIntervalContextIgnorantModelMixin, {}),
        (NonPredictionIntervalContextIgnorantModelMixin, {}),
        (PredictionIntervalContextRequiredModelMixin, {"prediction_size": 10}),
        (NonPredictionIntervalContextRequiredModelMixin, {"prediction_size": 10}),
    ],
)
@pytest.mark.parametrize("return_components", (True, False))
def test_model_mixins_calls_add_target_components_in_predict(mixin_constructor, return_components, call_params):
    with patch.multiple(mixin_constructor, __abstractmethods__=set()):
        ts = Mock()
        predict_ts = Mock(spec=TSDataset)
        mixin = mixin_constructor()
        mixin._predict = Mock(return_value=predict_ts)
        mixin._add_target_components = Mock()
        _ = mixin.predict(ts=ts, return_components=return_components, **call_params)

        mixin._add_target_components.assert_called_with(
            ts=ts,
            predictions=predict_ts,
            components_prediction_method=mixin._predict_components,
            return_components=return_components,
        )


@pytest.mark.parametrize("mixin_constructor", [PerSegmentModelMixin])
def test_make_prediction_segment_with_components(mixin_constructor, target_components_df, ts_without_target_components):
    mixin = mixin_constructor(base_model=Mock())
    target_components_df_model_format = TSDataset.to_flatten(target_components_df).drop(columns=["segment"])
    prediction_method = Mock(return_value=target_components_df_model_format)

    target_components_pred = mixin._make_predictions_segment(
        model=mixin._base_model,
        segment="1",
        df=ts_without_target_components.to_pandas(),
        prediction_method=prediction_method,
    )

    pd.testing.assert_frame_equal(
        target_components_pred.set_index(["timestamp", "segment"]),
        TSDataset.to_flatten(target_components_df).set_index(["timestamp", "segment"]),
    )


@pytest.mark.parametrize("mixin_constructor", [PerSegmentModelMixin, MultiSegmentModelMixin])
def test_make_components_prediction(mixin_constructor, target_components_df, ts_without_target_components):
    mixin = mixin_constructor(base_model=Mock())
    mixin.fit(ts_without_target_components)
    target_components_df_model_format = TSDataset.to_flatten(target_components_df).drop(columns=["segment"])
    prediction_method = Mock(return_value=target_components_df_model_format)

    target_components_pred = mixin._make_component_predictions(
        ts=ts_without_target_components, prediction_method=prediction_method
    )
    pd.testing.assert_frame_equal(target_components_pred, target_components_df)
