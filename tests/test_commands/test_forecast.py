from pathlib import Path
from subprocess import run
from tempfile import NamedTemporaryFile

import pandas as pd
import pytest


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
