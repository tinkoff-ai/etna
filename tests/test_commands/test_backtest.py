from pathlib import Path
from tempfile import NamedTemporaryFile
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pytest

from etna.commands import backtest
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
def base_backtest_yaml_path():
    tmp = NamedTemporaryFile("w")
    tmp.write(
        """
        n_folds: 3
        n_jobs: 3
        metrics:
          - _target_: etna.metrics.MAE
          - _target_: etna.metrics.MSE
          - _target_: etna.metrics.MAPE
          - _target_: etna.metrics.SMAPE
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


def test_dummy_run_with_exog(base_pipeline_yaml_path, base_backtest_yaml_path, base_timeseries_path, base_timeseries_exog_path):
    tmp_output = TemporaryDirectory()
    tmp_output_path = Path(tmp_output.name)
    backtest(
        config_path=base_pipeline_yaml_path,
        backtest_config_path=base_backtest_yaml_path,
        target_path=base_timeseries_path,
        freq="D",
        output_path=tmp_output_path,
        exog_path=base_timeseries_exog_path,
    )
    for file_name in ["metrics.csv", "forecast.csv", "info.csv"]:
        assert Path.exists(tmp_output_path / file_name)
