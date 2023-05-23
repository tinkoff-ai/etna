from pathlib import Path
from subprocess import run
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import pytest

from etna.commands.forecast_command import compute_horizon
from etna.commands.forecast_command import filter_forecast
from etna.commands.forecast_command import get_forecast_call_params
from etna.datasets import TSDataset


def test_dummy_run_with_exog(base_pipeline_yaml_path, base_timeseries_path, base_timeseries_exog_path):
    tmp_output = NamedTemporaryFile("w")
    tmp_output_path = Path(tmp_output.name)
    run(
        [
            "etna",
            "forecast",
            str(base_pipeline_yaml_path),
            str(base_timeseries_path),
            "D",
            str(tmp_output_path),
            str(base_timeseries_exog_path),
        ]
    )
    df_output = pd.read_csv(tmp_output_path)
    assert len(df_output) == 2 * 4


def test_omegaconf_run_with_exog(base_pipeline_omegaconf_path, base_timeseries_path, base_timeseries_exog_path):
    tmp_output = NamedTemporaryFile("w")
    tmp_output_path = Path(tmp_output.name)
    run(
        [
            "etna",
            "forecast",
            str(base_pipeline_omegaconf_path),
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
    run(["etna", "forecast", str(base_pipeline_yaml_path), str(base_timeseries_path), "D", str(tmp_output_path)])
    df_output = pd.read_csv(tmp_output_path)
    assert len(df_output) == 2 * 4


def test_run_with_predictive_intervals(
    base_pipeline_yaml_path, base_timeseries_path, base_timeseries_exog_path, base_forecast_omegaconf_path
):
    tmp_output = NamedTemporaryFile("w")
    tmp_output_path = Path(tmp_output.name)
    run(
        [
            "etna",
            "forecast",
            str(base_pipeline_yaml_path),
            str(base_timeseries_path),
            "D",
            str(tmp_output_path),
            str(base_timeseries_exog_path),
            str(base_forecast_omegaconf_path),
        ]
    )
    df_output = pd.read_csv(tmp_output_path)
    for q in [0.025, 0.975]:
        assert f"target_{q}" in df_output.columns


@pytest.mark.parametrize(
    "model_pipeline",
    [
        "elementary_linear_model_pipeline",
        "elementary_boosting_model_pipeline",
    ],
)
def test_forecast_use_exog_correct(
    model_pipeline, increasing_timeseries_path, increasing_timeseries_exog_path, request
):
    tmp_output = NamedTemporaryFile("w")
    tmp_output_path = Path(tmp_output.name)
    model_pipeline = request.getfixturevalue(model_pipeline)
    run(
        [
            "etna",
            "forecast",
            str(model_pipeline),
            str(increasing_timeseries_path),
            "D",
            str(tmp_output_path),
            str(increasing_timeseries_exog_path),
        ]
    )
    df_output = pd.read_csv(tmp_output_path)
    pd.testing.assert_series_equal(
        df_output["target"], pd.Series(data=[3.0, 3.0, 3.0], name="target"), check_less_precise=1
    )


@pytest.fixture
def ms_tsds():
    df = pd.DataFrame(
        {
            "timestamp": list(pd.date_range("2023-01-01", periods=4, freq="MS")) * 2,
            "segment": ["A"] * 4 + ["B"] * 4,
            "target": list(3 * np.arange(1, 5)) * 2,
        }
    )

    df = TSDataset.to_dataset(df=df)
    ts = TSDataset(df=df, freq="MS")
    return ts


@pytest.fixture
def pipeline_dummy_config():
    return {"horizon": 3}


@pytest.mark.parametrize(
    "params,expected",
    (
        ({"start_timestamp": "2021-09-10"}, {}),
        (
            {"prediction_interval": True, "n_folds": 3, "start_timestamp": "2021-09-10"},
            {"prediction_interval": True, "n_folds": 3},
        ),
        (
            {"prediction_interval": True, "n_folds": 3, "quantiles": [0.025, 0.975]},
            {"prediction_interval": True, "n_folds": 3, "quantiles": [0.025, 0.975]},
        ),
    ),
)
def test_drop_additional_forecast_params(params, expected):
    result = get_forecast_call_params(forecast_params=params)
    assert result == expected


@pytest.mark.parametrize("forecast_params", ({"start_timestamp": "2020-04-09"}, {"start_timestamp": "2019-04-10"}))
def test_compute_horizon_error(example_tsds, forecast_params, pipeline_dummy_config):
    with pytest.raises(ValueError, match="Parameter `start_timestamp` should greater than end of training dataset!"):
        compute_horizon(
            horizon=pipeline_dummy_config["horizon"], forecast_params=forecast_params, tsdataset=example_tsds
        )


@pytest.mark.parametrize(
    "forecast_params,tsdataset_name,expected",
    (
        ({"start_timestamp": "2020-04-10"}, "example_tsds", 3),
        ({"start_timestamp": "2020-04-12"}, "example_tsds", 5),
        ({"start_timestamp": "2020-02-01 02:00:00"}, "example_tsdf", 4),
        ({"start_timestamp": "2023-06-01"}, "ms_tsds", 4),
    ),
)
def test_compute_horizon(forecast_params, tsdataset_name, expected, request, pipeline_dummy_config):
    tsdataset = request.getfixturevalue(tsdataset_name)
    result = compute_horizon(
        horizon=pipeline_dummy_config["horizon"], forecast_params=forecast_params, tsdataset=tsdataset
    )
    assert result == expected


@pytest.mark.parametrize(
    "forecast_params,expected",
    (
        ({"start_timestamp": "2020-04-06"}, "2020-04-06"),
        ({}, "2020-01-01"),
    ),
)
def test_filter_forecast(forecast_params, expected, example_tsds):
    result = filter_forecast(forecast_ts=example_tsds, forecast_params=forecast_params)
    assert result.df.index.min() == pd.Timestamp(expected)


@pytest.mark.parametrize(
    "model_pipeline",
    [
        "elementary_linear_model_pipeline",
        "elementary_boosting_model_pipeline",
    ],
)
def test_forecast_start_timestamp(
    model_pipeline, base_timeseries_path, base_timeseries_exog_path, start_timestamp_forecast_omegaconf_path, request
):
    tmp_output = NamedTemporaryFile("w")
    tmp_output_path = Path(tmp_output.name)
    model_pipeline = request.getfixturevalue(model_pipeline)

    run(
        [
            "etna",
            "forecast",
            str(model_pipeline),
            str(base_timeseries_path),
            "D",
            str(tmp_output_path),
            str(base_timeseries_exog_path),
            str(start_timestamp_forecast_omegaconf_path),
        ]
    )
    df_output = pd.read_csv(tmp_output_path)

    assert len(df_output) == 3 * 2  # 3 predictions for 2 segments
    assert df_output["timestamp"].min() == "2021-09-10"  # start_timestamp
    assert not np.any(df_output.isna().values)
