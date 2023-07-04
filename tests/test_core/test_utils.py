import pathlib
import tempfile

import pandas as pd
import pytest

from etna.core import load
from etna.models import NaiveModel
from etna.pipeline import Pipeline
from etna.transforms import AddConstTransform


def test_load_fail_file_not_found():
    non_existent_path = pathlib.Path("archive.zip")
    with pytest.raises(FileNotFoundError):
        load(non_existent_path)


def test_load_ok():
    transform = AddConstTransform(in_column="target", value=10.0, inplace=False)
    with tempfile.TemporaryDirectory() as _temp_dir:
        temp_dir = pathlib.Path(_temp_dir)
        save_path = temp_dir / "transform.zip"
        transform.save(save_path)

        new_transform = load(save_path)

        assert type(new_transform) == type(transform)
        for attribute in ["in_column", "value", "inplace"]:
            assert getattr(new_transform, attribute) == getattr(transform, attribute)


def test_load_ok_with_params(example_tsds):
    pipeline = Pipeline(model=NaiveModel(), horizon=7)
    with tempfile.TemporaryDirectory() as _temp_dir:
        temp_dir = pathlib.Path(_temp_dir)
        save_path = temp_dir / "pipeline.zip"
        pipeline.fit(ts=example_tsds)
        pipeline.save(save_path)

        new_pipeline = load(save_path, ts=example_tsds)

        assert new_pipeline.ts is not None
        assert type(new_pipeline) == type(pipeline)
        pd.testing.assert_frame_equal(new_pipeline.ts.to_pandas(), example_tsds.to_pandas())
