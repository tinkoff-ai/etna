import json
import pathlib
from unittest.mock import MagicMock
from unittest.mock import patch
from zipfile import ZipFile

import dill
import pytest

from etna import SETTINGS

if SETTINGS.torch_required:
    import torch

from etna.models.mixins import MultiSegmentModelMixin
from etna.models.mixins import PerSegmentModelMixin
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
