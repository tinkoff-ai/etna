import json
import pathlib
import pickle
import zipfile
from copy import deepcopy
from unittest.mock import patch

import pandas as pd
import pytest

from etna.ensembles.mixins import SaveEnsembleMixin
from etna.ensembles.stacking_ensemble import StackingEnsemble
from etna.models import NaiveModel
from etna.pipeline import Pipeline

HORIZON = 7


def test_ensemble_invalid_pipelines_number(catboost_pipeline: Pipeline):
    """Test StackingEnsemble behavior in case of invalid pipelines number."""
    with pytest.raises(ValueError, match="At least two pipelines are expected."):
        _ = StackingEnsemble(pipelines=[catboost_pipeline])


def test_ensemble_get_horizon_pass(catboost_pipeline: Pipeline, prophet_pipeline: Pipeline):
    """Check that StackingEnsemble._get horizon works correctly in case of valid pipelines list."""
    horizon = StackingEnsemble._get_horizon(pipelines=[catboost_pipeline, prophet_pipeline])
    assert horizon == HORIZON


def test_ensemble_get_horizon_fail(catboost_pipeline: Pipeline, naive_pipeline: Pipeline):
    """Check that StackingEnsemble._get horizon works correctly in case of invalid pipelines list."""
    with pytest.raises(ValueError, match="All the pipelines should have the same horizon."):
        _ = StackingEnsemble._get_horizon(pipelines=[catboost_pipeline, naive_pipeline])


class Dummy(SaveEnsembleMixin):
    def __init__(self, a, b, ts, pipelines):
        self.a = a
        self.b = b
        self.ts = ts
        self.pipelines = pipelines


def test_save_mixin_save(example_tsds, tmp_path):
    pipelines = [Pipeline(model=NaiveModel(lag=1), horizon=HORIZON), Pipeline(model=NaiveModel(lag=2), horizon=HORIZON)]
    dummy = Dummy(a=1, b=2, ts=example_tsds, pipelines=pipelines)
    dir_path = pathlib.Path(tmp_path)
    path = dir_path / "dummy.zip"

    initial_dummy = deepcopy(dummy)
    dummy.save(path)

    with zipfile.ZipFile(path, "r") as archive:
        files = archive.namelist()
        assert sorted(files) == sorted(
            ["metadata.json", "object.pkl", "pipelines/00000000.zip", "pipelines/00000001.zip"]
        )

        with archive.open("metadata.json", "r") as input_file:
            metadata_bytes = input_file.read()
        metadata_str = metadata_bytes.decode("utf-8")
        metadata = json.loads(metadata_str)
        assert sorted(metadata.keys()) == ["class", "etna_version"]
        assert metadata["class"] == "tests.test_ensembles.test_mixins.Dummy"

        with archive.open("object.pkl", "r") as input_file:
            loaded_obj = pickle.load(input_file)
        assert loaded_obj.a == dummy.a
        assert loaded_obj.b == dummy.b

    # basic check that we didn't break dummy object itself
    assert dummy.a == initial_dummy.a
    assert pickle.dumps(dummy.ts) == pickle.dumps(initial_dummy.ts)
    assert len(dummy.pipelines) == len(initial_dummy.pipelines)


def test_save_mixin_load_fail_file_not_found():
    non_existent_path = pathlib.Path("archive.zip")
    with pytest.raises(FileNotFoundError):
        Dummy.load(non_existent_path)


def test_save_mixin_load_ok_no_ts(example_tsds, recwarn, tmp_path):
    lag_values = list(range(1, 11))
    pipelines = [Pipeline(model=NaiveModel(lag=lag), horizon=HORIZON) for lag in lag_values]
    dummy = Dummy(a=1, b=2, ts=example_tsds, pipelines=pipelines)
    dir_path = pathlib.Path(tmp_path)
    path = dir_path / "dummy.zip"

    dummy.save(path)
    loaded_dummy = Dummy.load(path)

    assert loaded_dummy.a == dummy.a
    assert loaded_dummy.b == dummy.b
    assert loaded_dummy.ts is None
    assert [pipeline.model.lag for pipeline in loaded_dummy.pipelines] == lag_values
    assert len(recwarn) == 0


def test_save_mixin_load_ok_with_ts(example_tsds, recwarn, tmp_path):
    lag_values = list(range(1, 11))
    pipelines = [Pipeline(model=NaiveModel(lag=lag), horizon=HORIZON) for lag in lag_values]
    dummy = Dummy(a=1, b=2, ts=example_tsds, pipelines=pipelines)
    dir_path = pathlib.Path(tmp_path)
    path = dir_path / "dummy.zip"

    dummy.save(path)
    loaded_dummy = Dummy.load(path, ts=example_tsds)

    assert loaded_dummy.a == dummy.a
    assert loaded_dummy.b == dummy.b
    assert loaded_dummy.ts is not example_tsds
    pd.testing.assert_frame_equal(loaded_dummy.ts.to_pandas(), dummy.ts.to_pandas())
    assert [pipeline.model.lag for pipeline in loaded_dummy.pipelines] == lag_values
    assert len(recwarn) == 0


@pytest.mark.parametrize(
    "save_version, load_version", [((1, 5, 0), (2, 5, 0)), ((2, 5, 0), (1, 5, 0)), ((1, 5, 0), (1, 3, 0))]
)
@patch("etna.core.mixins.get_etna_version")
def test_save_mixin_load_warning(get_version_mock, save_version, load_version, example_tsds, tmp_path):
    pipelines = [Pipeline(model=NaiveModel(lag=1), horizon=HORIZON), Pipeline(model=NaiveModel(lag=2), horizon=HORIZON)]
    dummy = Dummy(a=1, b=2, ts=example_tsds, pipelines=pipelines)
    dir_path = pathlib.Path(tmp_path)
    path = dir_path / "dummy.zip"

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
