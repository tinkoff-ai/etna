from pathlib import Path
from subprocess import run
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import pytest

from etna.commands import forecast
from etna.datasets import generate_ar_df
from etna.datasets.datasets_generation import generate_ar_df


@pytest.fixture
def base_pipeline_yaml_path():
    tmp = NamedTemporaryFile("w")
    tmp.write(
        """
        _target_: etna.pipeline.Pipeline
        horizon: 4
        model:
          _target_: etna.models.CatBoostModelMultiSegment
        transforms:
          - _target_: etna.transforms.LinearTrendTransform
            in_column: target
          - _target_: etna.transforms.SegmentEncoderTransform
        """
    )
    tmp.flush()
    yield Path(tmp.name)
    tmp.close()


@pytest.fixture
def base_timeseries_path():
    df = generate_ar_df(periods=100, start_time="2021-06-01", n_segments=2)
    tmp = NamedTemporaryFile("w")
    df.to_csv(tmp, index=False)
    tmp.flush()
    yield Path(tmp.name)
    tmp.close()


@pytest.fixture
def base_timeseries_exog_path():
    df_regressors = pd.DataFrame(
        {
            "timestamp": list(pd.date_range("2021-06-01", periods=120)) * 2,
            "regressor_1": np.arange(240),
            "regressor_2": np.arange(240) + 5,
            "segment": ["segment_0"] * 120 + ["segment_1"] * 120,
        }
    )
    tmp = NamedTemporaryFile("w")
    df_regressors.to_csv(tmp, index=False)
    tmp.flush()
    yield Path(tmp.name)
    tmp.close()


def test_dummy_run_with_exog(base_pipeline_yaml_path, base_timeseries_path, base_timeseries_exog_path):
    tmp_output = NamedTemporaryFile("w")
    tmp_output_path = Path(tmp_output.name)
    run(
        [
            "etna",
            str(base_pipeline_yaml_path),
            str(base_timeseries_path),
            "D",
            str(tmp_output_path),
            str(base_timeseries_exog_path),
        ]
    )
    df_output = pd.read_csv(tmp_output_path)
    assert len(df_output) == 2 * 4


def test_dummy_run(base_pipeline_yaml_path, base_timeseries_path):
    tmp_output = NamedTemporaryFile("w")
    tmp_output_path = Path(tmp_output.name)
    run(["etna", str(base_pipeline_yaml_path), str(base_timeseries_path), "D", str(tmp_output_path)])
    df_output = pd.read_csv(tmp_output_path)
    assert len(df_output) == 2 * 4
