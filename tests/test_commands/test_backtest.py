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


@pytest.fixture
def backtest_with_folds_estimation_yaml_path():
    tmp = NamedTemporaryFile("w")
    tmp.write(
        """
        n_folds: 200
        n_jobs: 4
        metrics:
          - _target_: etna.metrics.MAE
          - _target_: etna.metrics.MSE
          - _target_: etna.metrics.MAPE
          - _target_: etna.metrics.SMAPE
        estimate_n_folds: true
        """
    )
    tmp.flush()
    yield Path(tmp.name)
    tmp.close()


@pytest.fixture
def backtest_with_stride_yaml_path():
    tmp = NamedTemporaryFile("w")
    tmp.write(
        """
        n_folds: 3
        n_jobs: 4
        metrics:
          - _target_: etna.metrics.MAE
          - _target_: etna.metrics.MSE
          - _target_: etna.metrics.MAPE
          - _target_: etna.metrics.SMAPE
        estimate_n_folds: true
        stride: 100
        """
    )
    tmp.flush()
    yield Path(tmp.name)
    tmp.close()


@pytest.mark.parametrize("pipeline_path_name", ("base_pipeline_yaml_path", "base_ensemble_yaml_path"))
def test_dummy_run(pipeline_path_name, base_backtest_yaml_path, base_timeseries_path, request):
    tmp_output = TemporaryDirectory()
    tmp_output_path = Path(tmp_output.name)
    pipeline_path = request.getfixturevalue(pipeline_path_name)
    run(
        [
            "etna",
            "backtest",
            str(pipeline_path),
            str(base_backtest_yaml_path),
            str(base_timeseries_path),
            "D",
            str(tmp_output_path),
        ]
    )
    for file_name in ["metrics.csv", "forecast.csv", "info.csv"]:
        assert Path.exists(tmp_output_path / file_name)


@pytest.mark.parametrize("pipeline_path_name", ("base_pipeline_yaml_path", "base_ensemble_yaml_path"))
def test_dummy_run_with_exog(
    pipeline_path_name, base_backtest_yaml_path, base_timeseries_path, base_timeseries_exog_path, request
):
    tmp_output = TemporaryDirectory()
    tmp_output_path = Path(tmp_output.name)
    pipeline_path = request.getfixturevalue(pipeline_path_name)
    run(
        [
            "etna",
            "backtest",
            str(pipeline_path),
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
    assert all(x in forecast_df.columns for x in ["segment", "timestamp", "target"])
    assert len(forecast_df) == 24


@pytest.mark.parametrize(
    "pipeline_path_name,backtest_config_path_name,expected",
    (
        ("base_pipeline_with_context_size_yaml_path", "backtest_with_folds_estimation_yaml_path", 24),
        ("base_ensemble_yaml_path", "backtest_with_folds_estimation_yaml_path", 12),
        ("base_pipeline_with_context_size_yaml_path", "backtest_with_stride_yaml_path", 1),
    ),
)
def test_backtest_estimate_n_folds(
    pipeline_path_name, backtest_config_path_name, base_timeseries_path, expected, request
):
    backtest_config_path = request.getfixturevalue(backtest_config_path_name)
    pipeline_path = request.getfixturevalue(pipeline_path_name)

    tmp_output = TemporaryDirectory()
    tmp_output_path = Path(tmp_output.name)
    run(
        [
            "etna",
            "backtest",
            str(pipeline_path),
            str(backtest_config_path),
            str(base_timeseries_path),
            "D",
            str(tmp_output_path),
        ]
    )
    forecast_df = pd.read_csv(tmp_output_path / "forecast.csv")
    assert forecast_df["fold_number"].nunique() == expected
