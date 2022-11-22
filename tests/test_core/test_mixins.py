import json
import pathlib
import pickle
import tempfile
from unittest.mock import patch
from zipfile import ZipFile

import pytest

from etna.core.mixins import SaveMixin
from etna.core.mixins import get_etna_version


class Dummy(SaveMixin):
    def __init__(self, a, b):
        self.a = a
        self.b = b


def test_get_etna_version():
    version = get_etna_version()
    assert len(version) == 3


def test_save_mixin_save():
    with tempfile.TemporaryDirectory() as dir_path_str:
        dummy = Dummy(a=1, b=2)
        dir_path = pathlib.Path(dir_path_str)
        path = dir_path.joinpath("dummy.zip")

        dummy.save(path)

        with ZipFile(path, "r") as zip_file:
            files = zip_file.namelist()
            assert sorted(files) == ["metadata.json", "object.pkl"]

            with zip_file.open("metadata.json", "r") as input_file:
                metadata_bytes = input_file.read()
            metadata_str = metadata_bytes.decode("utf-8")
            metadata = json.loads(metadata_str)
            assert sorted(metadata.keys()) == ["class", "etna_version"]
            assert metadata["class"] == "tests.test_core.test_mixins.Dummy"

            with zip_file.open("object.pkl", "r") as input_file:
                loaded_dummy = pickle.load(input_file)
            assert loaded_dummy.a == dummy.a
            assert loaded_dummy.b == dummy.b


def test_save_mixin_load_ok(recwarn):
    with tempfile.TemporaryDirectory() as dir_path_str:
        dummy = Dummy(a=1, b=2)
        dir_path = pathlib.Path(dir_path_str)
        path = dir_path.joinpath("dummy.zip")

        dummy.save(path)
        loaded_dummy = Dummy.load(path)

        assert loaded_dummy.a == dummy.a
        assert loaded_dummy.b == dummy.b
        assert len(recwarn) == 0


@pytest.mark.parametrize(
    "save_version, load_version", [((1, 5, 0), (2, 5, 0)), ((2, 5, 0), (1, 5, 0)), ((1, 5, 0), (1, 3, 0))]
)
@patch("etna.core.mixins.get_etna_version")
def test_save_mixin_load_warning(get_version_mock, save_version, load_version):
    with tempfile.TemporaryDirectory() as dir_path_str:
        dummy = Dummy(a=1, b=2)
        dir_path = pathlib.Path(dir_path_str)
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
            _ = Dummy.load(path)
