from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import pytest

from etna.datasets import generate_ar_df


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
def base_pipeline_omegaconf_path():
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
          - _target_: etna.transforms.LagTransform
            in_column: target
            lags: "${shift:${horizon},[1, 2, 4]}"
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


@pytest.fixture
def base_forecast_omegaconf_path():
    tmp = NamedTemporaryFile("w")
    tmp.write(
        """
        prediction_interval: true
        quantiles: [0.025, 0.975]
        n_folds: 3
        """
    )
    tmp.flush()
    yield Path(tmp.name)
    tmp.close()
