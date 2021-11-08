from pathlib import Path
from subprocess import run
from tempfile import NamedTemporaryFile
from tempfile import TemporaryDirectory

import pandas as pd
import pytest


@pytest.fixture
def base_backtest_yaml_path():
    tmp = NamedTemporaryFile("w")
    tmp.write(
        """
        n_folds: 3
        n_jobs: ${n_folds}
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


def test_dummy_run(base_pipeline_yaml_path, base_backtest_yaml_path, base_timeseries_path):
    tmp_output = TemporaryDirectory()
    tmp_output_path = Path(tmp_output.name)
    run(
        [
            "etna",
            "backtest",
            str(base_pipeline_yaml_path),
            str(base_backtest_yaml_path),
            str(base_timeseries_path),
            "D",
            str(tmp_output_path),
        ]
    )
    for file_name in ["metrics.csv", "forecast.csv", "info.csv"]:
        assert Path.exists(tmp_output_path / file_name)


def test_dummy_run_with_exog(
    base_pipeline_yaml_path, base_backtest_yaml_path, base_timeseries_path, base_timeseries_exog_path
):
    tmp_output = TemporaryDirectory()
    tmp_output_path = Path(tmp_output.name)
    run(
        [
            "etna",
            "backtest",
            str(base_pipeline_yaml_path),
            str(base_backtest_yaml_path),
            str(base_timeseries_path),
            "D",
            str(tmp_output_path),
            str(base_timeseries_exog_path),
        ]
    )
    for file_name in ["metrics.csv", "forecast.csv", "info.csv"]:
        assert Path.exists(tmp_output_path / file_name)


def test_forecast_format(base_pipeline_yaml_path, base_backtest_yaml_path, base_timeseries_path):
    tmp_output = TemporaryDirectory()
    tmp_output_path = Path(tmp_output.name)
    run(
        [
            "etna",
            "backtest",
            str(base_pipeline_yaml_path),
            str(base_backtest_yaml_path),
            str(base_timeseries_path),
            "D",
            str(tmp_output_path),
        ]
    )
    forecast_df = pd.read_csv(tmp_output_path / "forecast.csv")
    assert all([x in forecast_df.columns for x in ["segment", "timestamp", "target"]])
    assert len(forecast_df) == 24
