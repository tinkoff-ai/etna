from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import pytest

from etna.datasets import TSDataset
from etna.datasets import generate_ar_df


@pytest.fixture
def base_pipeline_yaml_path():
    tmp = NamedTemporaryFile("w")
    tmp.write(
        """
        _target_: etna.pipeline.Pipeline
        horizon: 4
        model:
          _target_: etna.models.CatBoostMultiSegmentModel
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
def base_pipeline_with_context_size_yaml_path():
    tmp = NamedTemporaryFile("w")
    tmp.write(
        """
        _target_: etna.pipeline.Pipeline
        horizon: 4
        model:
          _target_: etna.models.CatBoostMultiSegmentModel
        transforms:
          - _target_: etna.transforms.LinearTrendTransform
            in_column: target
          - _target_: etna.transforms.SegmentEncoderTransform
        context_size: 1
        """
    )
    tmp.flush()
    yield Path(tmp.name)
    tmp.close()


@pytest.fixture
def base_ensemble_yaml_path():
    tmp = NamedTemporaryFile("w")
    tmp.write(
        """
        _target_: etna.ensembles.VotingEnsemble
        pipelines:
        - _target_: etna.pipeline.Pipeline
          horizon: 4
          model:
            _target_: etna.models.SeasonalMovingAverageModel
            seasonality: 4
            window: 1
          transforms: []
        - _target_: etna.pipeline.Pipeline
          horizon: 4
          model:
            _target_: etna.models.SeasonalMovingAverageModel
            seasonality: 7
            window: 2
          transforms: []
        - _target_: etna.pipeline.Pipeline
          horizon: 4
          model:
            _target_: etna.models.SeasonalMovingAverageModel
            seasonality: 7
            window: 7
          transforms: []
        context_size: 49
        """
    )
    tmp.flush()
    yield Path(tmp.name)
    tmp.close()


@pytest.fixture
def elementary_linear_model_pipeline():
    tmp = NamedTemporaryFile("w")
    tmp.write(
        """
        _target_: etna.pipeline.Pipeline
        horizon: 3
        model:
          _target_: etna.models.LinearPerSegmentModel
        """
    )
    tmp.flush()
    yield Path(tmp.name)
    tmp.close()


@pytest.fixture
def elementary_boosting_model_pipeline():
    tmp = NamedTemporaryFile("w")
    tmp.write(
        """
        _target_: etna.pipeline.Pipeline
        horizon: 3
        model:
          _target_: etna.models.CatBoostPerSegmentModel
        """
    )
    tmp.flush()
    yield Path(tmp.name)
    tmp.close()


@pytest.fixture
def increasing_timeseries_path():
    df = pd.DataFrame(
        {
            "timestamp": list(pd.date_range("2022-06-01", periods=10)),
            "target": list(range(10)),
            "segment": ["segment_0"] * 10,
        }
    )
    tmp = NamedTemporaryFile("w")
    df.to_csv(tmp, index=False)
    tmp.flush()
    yield Path(tmp.name)
    tmp.close()


@pytest.fixture
def increasing_timeseries_exog_path():
    df_regressors = pd.DataFrame(
        {
            "timestamp": list(pd.date_range("2022-06-01", periods=13)),
            "regressor_1": list(range(10)) + [3, 3, 3],
            "segment": ["segment_0"] * 13,
        }
    )
    tmp = NamedTemporaryFile("w")
    df_regressors.to_csv(tmp, index=False)
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
          _target_: etna.models.CatBoostMultiSegmentModel
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
def empty_ts():
    df = pd.DataFrame({"segment": [], "timestamp": [], "target": []})
    df = TSDataset.to_dataset(df=df)
    return TSDataset(df=df, freq="D")
