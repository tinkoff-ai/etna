import datetime
import json
import os
import pathlib
import tempfile
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from etna.datasets import TSDataset
from etna.ensembles import StackingEnsemble
from etna.loggers import LocalFileLogger
from etna.loggers import S3FileLogger
from etna.loggers import tslogger
from etna.metrics import MAE
from etna.metrics import MSE
from etna.metrics import SMAPE
from etna.models import NaiveModel
from etna.pipeline import Pipeline

DATETIME_FORMAT = "%Y-%m-%dT%H-%M-%S"


def test_local_file_logger_init_new_dir():
    """Test that LocalFileLogger creates subfolder during init."""
    with tempfile.TemporaryDirectory() as dirname:
        assert len(os.listdir(dirname)) == 0
        _ = LocalFileLogger(experiments_folder=dirname)
        assert len(os.listdir(dirname)) == 1


def test_local_file_logger_save_config():
    """Test that LocalFileLogger creates folder with config during init."""
    with tempfile.TemporaryDirectory() as dirname:
        cur_dir = pathlib.Path(dirname)
        example_config = {"key": "value"}
        _ = LocalFileLogger(experiments_folder=dirname, config=example_config)
        experiment_folder_name = os.listdir(dirname)[0]
        experiment_folder = cur_dir.joinpath(experiment_folder_name)
        assert len(os.listdir(experiment_folder)) == 1
        with open(experiment_folder.joinpath("config.json")) as inf:
            read_config = json.load(inf)
        assert read_config == example_config


def test_local_file_logger_start_experiment():
    """Test that LocalFileLogger creates new subfolder according to the parameters."""
    with tempfile.TemporaryDirectory() as dirname:
        cur_dir = pathlib.Path(dirname)
        # get rid of seconds fractions
        start_datetime = datetime.datetime.strptime(datetime.datetime.now().strftime(DATETIME_FORMAT), DATETIME_FORMAT)
        logger = LocalFileLogger(experiments_folder=dirname)
        experiment_folder_name = os.listdir(dirname)[0]
        experiment_folder = cur_dir.joinpath(experiment_folder_name)
        # get rid of seconds fractions
        end_datetime = datetime.datetime.strptime(datetime.datetime.now().strftime(DATETIME_FORMAT), DATETIME_FORMAT)

        folder_creation_datetime = datetime.datetime.strptime(experiment_folder_name, DATETIME_FORMAT)
        assert end_datetime >= folder_creation_datetime >= start_datetime
        assert len(os.listdir(experiment_folder)) == 0

        logger.start_experiment(job_type="test", group="1")
        assert len(os.listdir(experiment_folder)) == 1

        assert experiment_folder.joinpath("test").joinpath("1").exists()


def test_local_file_logger_fail_save_table():
    """Test that LocalFileLogger can't save table before starting the experiment."""
    with tempfile.TemporaryDirectory() as dirname:
        logger = LocalFileLogger(experiments_folder=dirname)

        example_df = pd.DataFrame({"keys": [1, 2, 3], "values": ["1", "2", "3"]})
        with pytest.raises(ValueError, match="You should start experiment before"):
            logger._save_table(example_df, "example")


def test_local_file_logger_save_table():
    """Test that LocalFileLogger saves table after starting the experiment."""
    with tempfile.TemporaryDirectory() as dirname:
        cur_dir = pathlib.Path(dirname)
        logger = LocalFileLogger(experiments_folder=dirname, gzip=False)
        experiment_folder_name = os.listdir(dirname)[0]
        experiment_folder = cur_dir.joinpath(experiment_folder_name)

        logger.start_experiment(job_type="example", group="example")
        example_df = pd.DataFrame({"keys": [1, 2, 3], "values": ["first", "second", "third"]})
        logger._save_table(example_df, "example")

        experiment_subfolder = experiment_folder.joinpath("example").joinpath("example")
        assert "example.csv" in os.listdir(experiment_subfolder)

        read_example_df = pd.read_csv(experiment_subfolder.joinpath("example.csv"))
        assert np.all(read_example_df == example_df)


def test_local_file_logger_fail_save_dict():
    """Test that LocalFileLogger can't save dict before starting the experiment."""
    with tempfile.TemporaryDirectory() as dirname:
        logger = LocalFileLogger(experiments_folder=dirname)

        example_dict = {"keys": [1, 2, 3], "values": ["first", "second", "third"]}
        with pytest.raises(ValueError, match="You should start experiment before"):
            logger._save_dict(example_dict, "example")


def test_local_file_logger_save_dict():
    """Test that LocalFileLogger saves dict after starting the experiment."""
    with tempfile.TemporaryDirectory() as dirname:
        cur_dir = pathlib.Path(dirname)
        logger = LocalFileLogger(experiments_folder=dirname)
        experiment_folder_name = os.listdir(dirname)[0]
        experiment_folder = cur_dir.joinpath(experiment_folder_name)

        logger.start_experiment(job_type="example", group="example")
        example_dict = {"keys": [1, 2, 3], "values": ["first", "second", "third"]}
        logger._save_dict(example_dict, "example")

        experiment_subfolder = experiment_folder.joinpath("example").joinpath("example")
        assert "example.json" in os.listdir(experiment_subfolder)

        with open(experiment_subfolder.joinpath("example.json")) as inf:
            read_example_dict = json.load(inf)
        assert read_example_dict == example_dict


def test_base_file_logger_log_backtest_run(example_tsds: TSDataset):
    """Test that BaseLogger correctly works in log_backtest_run on LocalFileLogger example."""
    with tempfile.TemporaryDirectory() as dirname:
        cur_dir = pathlib.Path(dirname)
        logger = LocalFileLogger(experiments_folder=dirname, gzip=False)
        experiment_folder_name = os.listdir(dirname)[0]
        experiment_folder = cur_dir.joinpath(experiment_folder_name)

        idx = tslogger.add(logger)
        metrics = [MAE(), MSE(), SMAPE()]
        pipeline = Pipeline(model=NaiveModel(), horizon=10)
        n_folds = 5
        pipeline.backtest(ts=example_tsds, metrics=metrics, n_jobs=1, n_folds=n_folds)

        for fold_number in range(n_folds):
            fold_folder = experiment_folder.joinpath("crossval").joinpath(str(fold_number))
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
        logger = LocalFileLogger(experiments_folder=dirname, gzip=False)
        experiment_folder_name = os.listdir(dirname)[0]
        experiment_folder = cur_dir.joinpath(experiment_folder_name)

        idx = tslogger.add(logger)
        metrics = [MAE(), MSE(), SMAPE()]
        pipeline = Pipeline(model=NaiveModel(), horizon=10)
        n_folds = 5
        metrics_df, forecast_df, fold_info_df = pipeline.backtest(
            ts=example_tsds, metrics=metrics, n_jobs=1, n_folds=n_folds, aggregate_metrics=aggregate_metrics
        )

        crossval_results_folder = experiment_folder.joinpath("crossval_results").joinpath("all")

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


def test_local_file_logger_with_stacking_ensemble(example_df):
    """Test that LocalFileLogger correctly works in with stacking."""
    with tempfile.TemporaryDirectory() as dirname:
        cur_dir = pathlib.Path(dirname)
        logger = LocalFileLogger(experiments_folder=dirname, gzip=False)

        idx = tslogger.add(logger)
        example_df = TSDataset.to_dataset(example_df)
        example_df = TSDataset(example_df, freq="1H")
        ensemble_pipeline = StackingEnsemble(
            pipelines=[
                Pipeline(
                    model=NaiveModel(lag=10),
                    transforms=[],
                    horizon=5,
                ),
                Pipeline(
                    model=NaiveModel(lag=10),
                    transforms=[],
                    horizon=5,
                ),
            ]
        )
        n_folds = 5

        _ = ensemble_pipeline.backtest(example_df, metrics=[MAE()], n_jobs=4, n_folds=n_folds)

        assert len(list(cur_dir.iterdir())) == 1, "we've run one experiment"

        current_experiment_dir = list(cur_dir.iterdir())[0]
        assert len(list(current_experiment_dir.iterdir())) == 2, "crossval and crossval_results folders"

        assert (
            len(list((current_experiment_dir / "crossval").iterdir())) == n_folds
        ), "crossval should have `n_folds` runs"

        tslogger.remove(idx)


def test_local_file_logger_with_empirical_prediction_interval(example_df):
    """Test that LocalFileLogger correctly works in with empirical predicition intervals via backtest."""
    with tempfile.TemporaryDirectory() as dirname:
        cur_dir = pathlib.Path(dirname)
        logger = LocalFileLogger(experiments_folder=dirname, gzip=False)

        idx = tslogger.add(logger)
        example_df = TSDataset.to_dataset(example_df)
        example_df = TSDataset(example_df, freq="1H")
        pipe = Pipeline(model=NaiveModel(), transforms=[], horizon=2)
        n_folds = 5

        _ = pipe.backtest(
            example_df,
            metrics=[MAE()],
            n_jobs=4,
            n_folds=n_folds,
            forecast_params={"prediction_interval": True},
        )

        assert len(list(cur_dir.iterdir())) == 1, "we've run one experiment"

        current_experiment_dir = list(cur_dir.iterdir())[0]
        assert len(list(current_experiment_dir.iterdir())) == 2, "crossval and crossval_results folders"

        assert (
            len(list((current_experiment_dir / "crossval").iterdir())) == n_folds
        ), "crossval should have `n_folds` runs"

        tslogger.remove(idx)


def test_s3_file_logger_fail_init_endpoint_url(monkeypatch):
    """Test that S3FileLogger can't be created without setting 'endpoint_url' environment variable."""
    monkeypatch.delenv("endpoint_url", raising=False)
    monkeypatch.setenv("aws_access_key_id", "example")
    monkeypatch.setenv("aws_secret_access_key", "example")
    with pytest.raises(OSError, match="Environment variable `endpoint_url` should be specified"):
        _ = S3FileLogger(bucket="example", experiments_folder="experiments_folder")


def test_s3_file_logger_fail_init_aws_access_key_id(monkeypatch):
    """Test that S3FileLogger can't be created without setting 'aws_access_key_id' environment variable."""
    monkeypatch.setenv("endpoint_url", "https://s3.example.com")
    monkeypatch.delenv("aws_access_key_id", raising=False)
    monkeypatch.setenv("aws_secret_access_key", "example")
    with pytest.raises(OSError, match="Environment variable `aws_access_key_id` should be specified"):
        _ = S3FileLogger(bucket="example", experiments_folder="experiments_folder")


def test_s3_file_logger_fail_init_aws_secret_access_key(monkeypatch):
    """Test that S3FileLogger can't be created without setting 'aws_secret_access_key' environment variable."""
    monkeypatch.setenv("endpoint_url", "https://s3.example.com")
    monkeypatch.setenv("aws_access_key_id", "example")
    monkeypatch.delenv("aws_secret_access_key", raising=False)
    with pytest.raises(OSError, match="Environment variable `aws_secret_access_key` should be specified"):
        _ = S3FileLogger(bucket="example", experiments_folder="experiments_folder")


@mock.patch("etna.loggers.S3FileLogger._check_bucket", return_value=None)
@mock.patch("etna.loggers.S3FileLogger._get_s3_client", return_value=None)
def test_s3_file_logger_fail_save_table(check_bucket_fn, get_s3_client_fn):
    """Test that S3FileLogger can't save table before starting the experiment."""
    logger = S3FileLogger(bucket="example", experiments_folder="experiments_folder")

    example_df = pd.DataFrame({"keys": [1, 2, 3], "values": ["first", "second", "third"]})
    with pytest.raises(ValueError, match="You should start experiment before"):
        logger._save_table(example_df, "example")


@mock.patch("etna.loggers.S3FileLogger._check_bucket", return_value=None)
@mock.patch("etna.loggers.S3FileLogger._get_s3_client", return_value=None)
def test_s3_file_logger_fail_save_dict(check_bucket_fn, get_s3_client_fn):
    """Test that S3FileLogger can't save dict before starting the experiment."""
    logger = S3FileLogger(bucket="example", experiments_folder="experiments_folder")

    example_dict = {"keys": [1, 2, 3], "values": ["first", "second", "third"]}
    with pytest.raises(ValueError, match="You should start experiment before"):
        logger._save_dict(example_dict, "example")


@pytest.mark.skip
def test_s3_file_logger_save_table():
    """Test that S3FileLogger saves table after starting the experiment.

    This test is optional and requires environment variable 'etna_test_s3_bucket' to be set.
    """
    bucket = os.getenv("etna_test_s3_bucket")
    if bucket is None:
        raise OSError("To perform this test you should set 'etna_test_s3_bucket' environment variable first")
    experiments_folder = "s3_logger_test"
    logger = S3FileLogger(bucket=bucket, experiments_folder=experiments_folder, gzip=False)
    logger.start_experiment(job_type="test_simple", group="1")
    example_df = pd.DataFrame({"keys": [1, 2, 3], "values": ["first", "second", "third"]})
    logger._save_table(example_df, "example")

    list_objects = logger.s3_client.list_objects(Bucket=bucket)["Contents"]
    test_files = [filename["Key"] for filename in list_objects if filename["Key"].startswith(experiments_folder)]
    assert len(test_files) > 0
    key = max(test_files, key=lambda x: datetime.datetime.strptime(x.split("/")[1], DATETIME_FORMAT))

    with tempfile.NamedTemporaryFile() as ouf:
        logger.s3_client.download_file(Bucket=bucket, Key=key, Filename=ouf.name)
        read_example_df = pd.read_csv(ouf.name)
    assert np.all(read_example_df == example_df)


@pytest.mark.skip
def test_s3_file_logger_save_dict():
    """Test that S3FileLogger saves dict after starting the experiment.

    This test is optional and requires environment variable 'etna_test_s3_bucket' to be set.
    """
    bucket = os.environ["etna_test_s3_bucket"]
    experiments_folder = "s3_logger_test"
    logger = S3FileLogger(bucket=bucket, experiments_folder=experiments_folder, gzip=False)
    logger.start_experiment(job_type="test_simple", group="1")
    example_dict = {"keys": [1, 2, 3], "values": ["first", "second", "third"]}
    logger._save_dict(example_dict, "example")

    list_objects = logger.s3_client.list_objects(Bucket=bucket)["Contents"]
    test_files = [filename["Key"] for filename in list_objects if filename["Key"].startswith(experiments_folder)]
    assert len(test_files) > 0
    key = max(test_files, key=lambda x: datetime.datetime.strptime(x.split("/")[1], DATETIME_FORMAT))

    with tempfile.NamedTemporaryFile(delete=False) as ouf:
        logger.s3_client.download_file(Bucket=bucket, Key=key, Filename=ouf.name)
        cur_path = ouf.name
    with open(cur_path, "r") as inf:
        read_example_dict = json.load(inf)
    assert read_example_dict == example_dict
