import json
import os
import pathlib
import tempfile

import numpy as np
import pandas as pd
import pytest

from etna.datasets import TSDataset
from etna.loggers import LocalFileLogger
from etna.loggers import tslogger
from etna.metrics import MAE
from etna.metrics import MSE
from etna.metrics import SMAPE
from etna.models import NaiveModel
from etna.pipeline import Pipeline


def test_local_file_logger_fail_init():
    """Test that LocalFileLogger can't be created with wrong experiment_folder."""
    with pytest.raises(ValueError, match="Folder non-existent-dir doesn't exist"):
        _ = LocalFileLogger("non-existent-dir")


def test_local_file_logger_init_new_dir():
    """Test that LocalFileLogger creates subfolder during init."""
    with tempfile.TemporaryDirectory() as dirname:
        assert len(os.listdir(dirname)) == 0
        _ = LocalFileLogger(experiments_folder=dirname)
        assert len(os.listdir(dirname)) == 1


def test_local_file_logger_start_experiment():
    """Test that LocalFileLogger creates new subfolder according to the parameters."""
    with tempfile.TemporaryDirectory() as dirname:
        cur_dir = pathlib.Path(dirname)
        logger = LocalFileLogger(experiments_folder=dirname)
        experiment_folder = os.listdir(dirname)[0]
        assert len(os.listdir(cur_dir.joinpath(experiment_folder))) == 0
        logger.start_experiment(job_type="test", group="1")
        assert len(os.listdir(cur_dir.joinpath(experiment_folder))) == 1
        filename = os.listdir(cur_dir.joinpath(experiment_folder))[0]
        assert filename == f"test_1"


def test_local_file_logger_log_backtest_run(example_tsds: TSDataset):
    """Test that BaseLogger correclty works in log_backtest_run on LocalFileLogger example."""
    with tempfile.TemporaryDirectory() as dirname:
        cur_dir = pathlib.Path(dirname)
        logger = LocalFileLogger(experiments_folder=dirname)
        experiment_folder_name = os.listdir(dirname)[0]
        experiment_folder = cur_dir.joinpath(experiment_folder_name)

        idx = tslogger.add(logger)
        metrics = [MAE(), MSE(), SMAPE()]
        pipeline = Pipeline(model=NaiveModel(), horizon=10)
        n_folds = 5
        pipeline.backtest(ts=example_tsds, metrics=metrics, n_jobs=1, n_folds=n_folds)

        for fold_number in range(n_folds):
            fold_folder = experiment_folder.joinpath(f"crossval_{fold_number}")
            assert "metrics.csv" in os.listdir(fold_folder)
            assert "forecast.csv" in os.listdir(fold_folder)
            assert "test.csv" in os.listdir(fold_folder)

            # check metrics summary
            with open(fold_folder.joinpath("metrics_summary.json"), "r") as inf:
                metrics_summary = json.load(inf)

            statistic_keys = [
                "median",
                "mean",
                "std",
                "percentile_5",
                "percentile_25",
                "percentile_75",
                "percentile_95",
            ]
            assert len(metrics_summary.keys()) == len(metrics) * len(statistic_keys)

    tslogger.remove(idx)


@pytest.mark.parametrize("aggregate_metrics", [True, False])
def test_base_file_logger_log_backtest_metrics(example_tsds: TSDataset, aggregate_metrics: bool):
    """Test that BaseFileLogger correctly works in log_backtest_metrics on LocaFileLogger example."""
    with tempfile.TemporaryDirectory() as dirname:
        cur_dir = pathlib.Path(dirname)
        logger = LocalFileLogger(experiments_folder=dirname)
        experiment_folder_name = os.listdir(dirname)[0]
        experiment_folder = cur_dir.joinpath(experiment_folder_name)

        idx = tslogger.add(logger)
        metrics = [MAE(), MSE(), SMAPE()]
        pipeline = Pipeline(model=NaiveModel(), horizon=10)
        n_folds = 5
        metrics_df, forecast_df, fold_info_df = pipeline.backtest(
            ts=example_tsds, metrics=metrics, n_jobs=1, n_folds=n_folds, aggregate_metrics=aggregate_metrics
        )

        crossval_results_folder = experiment_folder.joinpath("crossval_results_all")

        # check metrics_df
        metrics_df = metrics_df.reset_index(drop=True)
        metrics_df_saved = pd.read_csv(crossval_results_folder.joinpath("metrics.csv"))
        assert np.all(metrics_df_saved["segment"] == metrics_df["segment"])
        assert np.allclose(metrics_df_saved.drop(columns=["segment"]), metrics_df.drop(columns=["segment"]))

        # check forecast_df
        forecast_df = TSDataset.to_flatten(forecast_df)
        forecast_df_saved = pd.read_csv(
            crossval_results_folder.joinpath("forecast.csv"), parse_dates=["timestamp"], infer_datetime_format=True
        )
        assert np.all(
            forecast_df_saved[["timestamp", "fold_number", "segment"]]
            == forecast_df[["timestamp", "fold_number", "segment"]]
        )
        assert np.allclose(forecast_df_saved["target"], forecast_df["target"])

        # check fold_info_df
        fold_info_df = fold_info_df.reset_index(drop=True)
        fold_info_df_saved = pd.read_csv(
            crossval_results_folder.joinpath("fold_info.csv"),
            parse_dates=["train_start_time", "train_end_time", "test_start_time", "test_end_time"],
            infer_datetime_format=True,
        )
        assert np.all(fold_info_df_saved == fold_info_df)

        # check metrics summary
        with open(crossval_results_folder.joinpath("metrics_summary.json"), "r") as inf:
            metrics_summary = json.load(inf)

        statistic_keys = ["median", "mean", "std", "percentile_5", "percentile_25", "percentile_75", "percentile_95"]
        assert len(metrics_summary.keys()) == len(metrics) * len(statistic_keys)

    tslogger.remove(idx)
