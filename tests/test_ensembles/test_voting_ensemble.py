from copy import deepcopy
from typing import List
from typing import Optional
from typing import Union
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from joblib import Parallel
from joblib import delayed
from typing_extensions import Literal

from etna.datasets import TSDataset
from etna.ensembles.voting_ensemble import VotingEnsemble
from etna.metrics import MAE
from etna.pipeline import Pipeline

HORIZON = 7


@pytest.mark.parametrize(
    "weights",
    (None, [0.2, 0.3, 0.5], "auto"),
)
def test_validate_weights_pass(
    weights: Optional[Union[List[float], Literal["auto"]]],
):
    """Check that VotingEnsemble._validate_weights validate weights correctly in case of valid args sets."""
    VotingEnsemble._validate_weights(weights=weights, pipelines_number=3)


def test_validate_weights_fail():
    """Check that VotingEnsemble._validate_weights validate weights correctly in case of invalid args sets."""
    with pytest.raises(ValueError, match="Weights size should be equal to pipelines number."):
        _ = VotingEnsemble._validate_weights(weights=[0.3, 0.4, 0.3], pipelines_number=2)


@pytest.mark.parametrize(
    "weights,pipelines_number,expected",
    ((None, 5, [0.2, 0.2, 0.2, 0.2, 0.2]), ([0.2, 0.3, 0.5], 3, [0.2, 0.3, 0.5]), ([1, 1, 2], 3, [0.25, 0.25, 0.5])),
)
def test_process_weights(
    naive_pipeline_1: Pipeline,
    weights: Optional[List[float]],
    pipelines_number: int,
    expected: List[float],
):
    """Check that _process_weights processes weights correctly."""
    ensemble = VotingEnsemble(pipelines=[naive_pipeline_1 for _ in range(pipelines_number)], weights=weights)
    result = ensemble._process_weights()
    assert isinstance(result, list)
    assert result == expected


def test_process_weights_auto(example_tsdf: TSDataset, naive_pipeline_1: Pipeline, naive_pipeline_2: Pipeline):
    """Check that _process_weights processes weights correctly in "auto" mode."""
    ensemble = VotingEnsemble(pipelines=[naive_pipeline_1, naive_pipeline_2], weights="auto")
    ensemble.ts = example_tsdf
    result = ensemble._process_weights()
    assert isinstance(result, list)
    assert result[0] > result[1]


@pytest.mark.parametrize(
    "weights",
    ((None, [0.2, 0.3], "auto")),
)
def test_fit_interface(
    example_tsdf: TSDataset,
    weights: Optional[Union[List[float], Literal["auto"]]],
    naive_pipeline_1: Pipeline,
    naive_pipeline_2: Pipeline,
):
    """Check that fit saves processes weights."""
    ensemble = VotingEnsemble(pipelines=[naive_pipeline_1, naive_pipeline_2], weights=weights)
    ensemble.fit(example_tsdf)
    result = ensemble.processed_weights
    assert isinstance(result, list)
    assert len(result) == 2


def test_forecast_interface(example_tsds: TSDataset, catboost_pipeline: Pipeline, prophet_pipeline: Pipeline):
    """Check that VotingEnsemble.forecast returns TSDataset of correct length."""
    ensemble = VotingEnsemble(pipelines=[catboost_pipeline, prophet_pipeline])
    ensemble.fit(ts=example_tsds)
    forecast = ensemble.forecast()
    assert isinstance(forecast, TSDataset)
    assert len(forecast.df) == HORIZON


def test_forecast_prediction_interval_interface(example_tsds, naive_pipeline_1, naive_pipeline_2):
    """Test the forecast interface with prediction intervals."""
    ensemble = VotingEnsemble(pipelines=[naive_pipeline_1, naive_pipeline_2], weights=[1, 3])
    ensemble.fit(example_tsds)
    forecast = ensemble.forecast(prediction_interval=True, quantiles=[0.025, 0.975])
    for segment in forecast.segments:
        segment_slice = forecast[:, segment, :][segment]
        assert {"target_0.025", "target_0.975", "target"}.issubset(segment_slice.columns)
        assert (segment_slice["target_0.975"] - segment_slice["target_0.025"] >= 0).all()


def test_predict_interface(example_tsds: TSDataset, catboost_pipeline: Pipeline, prophet_pipeline: Pipeline):
    """Check that VotingEnsemble.predict returns TSDataset of correct length."""
    ensemble = VotingEnsemble(pipelines=[catboost_pipeline, prophet_pipeline])
    ensemble.fit(ts=example_tsds)
    start_idx = 20
    end_idx = 30
    prediction = ensemble.predict(
        ts=example_tsds, start_timestamp=example_tsds.index[start_idx], end_timestamp=example_tsds.index[end_idx]
    )
    assert isinstance(prediction, TSDataset)
    assert len(prediction.df) == end_idx - start_idx + 1


def test_vote_default_weights(simple_df: TSDataset, naive_pipeline_1: Pipeline, naive_pipeline_2: Pipeline):
    """Check that VotingEnsemble gets average during vote."""
    ensemble = VotingEnsemble(pipelines=[naive_pipeline_1, naive_pipeline_2])
    ensemble.fit(ts=simple_df)
    forecasts = Parallel(n_jobs=ensemble.n_jobs, backend="multiprocessing", verbose=11)(
        delayed(ensemble._forecast_pipeline)(pipeline=pipeline) for pipeline in ensemble.pipelines
    )
    forecast = ensemble._vote(forecasts=forecasts)
    np.testing.assert_array_equal(forecast[:, "A", "target"].values, [47.5, 48, 47.5, 48, 47.5, 48, 47.5])
    np.testing.assert_array_equal(forecast[:, "B", "target"].values, [11, 12, 11, 12, 11, 12, 11])


def test_vote_custom_weights(simple_df: TSDataset, naive_pipeline_1: Pipeline, naive_pipeline_2: Pipeline):
    """Check that VotingEnsemble gets average during vote."""
    ensemble = VotingEnsemble(pipelines=[naive_pipeline_1, naive_pipeline_2], weights=[1, 3])
    ensemble.fit(ts=simple_df)
    forecasts = Parallel(n_jobs=ensemble.n_jobs, backend="multiprocessing", verbose=11)(
        delayed(ensemble._forecast_pipeline)(pipeline=pipeline) for pipeline in ensemble.pipelines
    )
    forecast = ensemble._vote(forecasts=forecasts)
    np.testing.assert_array_equal(forecast[:, "A", "target"].values, [47.25, 48, 47.25, 48, 47.25, 48, 47.25])
    np.testing.assert_array_equal(forecast[:, "B", "target"].values, [10.5, 12, 10.5, 12, 10.5, 12, 10.5])


def test_forecast_calls_vote(example_tsds: TSDataset, naive_pipeline_1: Pipeline, naive_pipeline_2: Pipeline):
    ensemble = VotingEnsemble(pipelines=[naive_pipeline_1, naive_pipeline_2])
    ensemble.fit(ts=example_tsds)
    ensemble._vote = MagicMock()

    result = ensemble._forecast()

    ensemble._vote.assert_called_once()
    assert result == ensemble._vote.return_value


def test_predict_calls_vote(example_tsds: TSDataset, naive_pipeline_1: Pipeline, naive_pipeline_2: Pipeline):
    ensemble = VotingEnsemble(pipelines=[naive_pipeline_1, naive_pipeline_2])
    ensemble.fit(ts=example_tsds)
    ensemble._vote = MagicMock()

    result = ensemble._predict(
        ts=example_tsds,
        start_timestamp=example_tsds.index[20],
        end_timestamp=example_tsds.index[30],
        prediction_interval=False,
        quantiles=(),
    )

    ensemble._vote.assert_called_once()
    assert result == ensemble._vote.return_value


@pytest.mark.long_1
def test_multiprocessing_ensembles(
    simple_df: TSDataset,
    catboost_pipeline: Pipeline,
    prophet_pipeline: Pipeline,
    naive_pipeline_1: Pipeline,
    naive_pipeline_2: Pipeline,
):
    """Check that VotingEnsemble works the same in case of multi and single jobs modes."""
    pipelines = [catboost_pipeline, prophet_pipeline, naive_pipeline_1, naive_pipeline_2]
    single_jobs_ensemble = VotingEnsemble(pipelines=deepcopy(pipelines), n_jobs=1)
    multi_jobs_ensemble = VotingEnsemble(pipelines=deepcopy(pipelines), n_jobs=3)

    single_jobs_ensemble.fit(ts=deepcopy(simple_df))
    multi_jobs_ensemble.fit(ts=deepcopy(simple_df))

    single_jobs_forecast = single_jobs_ensemble.forecast()
    multi_jobs_forecast = multi_jobs_ensemble.forecast()

    assert (single_jobs_forecast.df == multi_jobs_forecast.df).all().all()


@pytest.mark.long_1
@pytest.mark.parametrize("n_jobs", (1, 5))
def test_backtest(voting_ensemble_pipeline: VotingEnsemble, example_tsds: TSDataset, n_jobs: int):
    """Check that backtest works with VotingEnsemble."""
    results = voting_ensemble_pipeline.backtest(ts=example_tsds, metrics=[MAE()], n_jobs=n_jobs, n_folds=3)
    for df in results:
        assert isinstance(df, pd.DataFrame)
