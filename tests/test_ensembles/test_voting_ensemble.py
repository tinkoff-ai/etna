from copy import deepcopy
from typing import List
from typing import Optional

import numpy as np
import pandas as pd
import pytest

from etna.datasets import TSDataset
from etna.ensembles.voting_ensemble import VotingEnsemble
from etna.metrics import MAE
from etna.pipeline import Pipeline

HORIZON = 7


@pytest.mark.parametrize(
    "weights,pipelines_number,expected",
    ((None, 5, [0.2, 0.2, 0.2, 0.2, 0.2]), ([0.2, 0.3, 0.5], 3, [0.2, 0.3, 0.5]), ([1, 1, 2], 3, [0.25, 0.25, 0.5])),
)
def test_process_weights_pass(
    weights: Optional[List[float]],
    pipelines_number: int,
    expected: List[float],
    catboost_pipeline: Pipeline,
    prophet_pipeline: Pipeline,
):
    """Check that VotingEnsemble._process_weights processes weights correctly in case of valid args sets."""
    result = VotingEnsemble._process_weights(weights=weights, pipelines_number=pipelines_number)
    assert isinstance(result, list)
    assert all([x == y for x, y in zip(result, expected)])


def test_process_weights_fail():
    """Check that VotingEnsemble._process_weights processes weights correctly in case of invalid args sets."""
    with pytest.raises(ValueError, match="Weights size should be equal to pipelines number."):
        _ = VotingEnsemble._process_weights(weights=[0.3, 0.4, 0.3], pipelines_number=2)


def test_forecast_interface(example_tsds: TSDataset, catboost_pipeline: Pipeline, prophet_pipeline: Pipeline):
    """Check that VotingEnsemble.forecast returns TSDataset of correct length."""
    ensemble = VotingEnsemble(pipelines=[catboost_pipeline, prophet_pipeline])
    ensemble.fit(ts=example_tsds)
    forecast = ensemble.forecast()
    assert isinstance(forecast, TSDataset)
    assert len(forecast.df) == HORIZON


def test_forecast_values_default_weights(simple_df: TSDataset, naive_pipeline_1: Pipeline, naive_pipeline_2: Pipeline):
    """Check that VotingEnsemble gets average."""
    ensemble = VotingEnsemble(pipelines=[naive_pipeline_1, naive_pipeline_2])
    ensemble.fit(ts=simple_df)
    forecast = ensemble.forecast()
    np.testing.assert_array_equal(forecast[:, "A", "target"].values, [47.5, 48, 47.5, 48, 47.5, 48, 47.5])
    np.testing.assert_array_equal(forecast[:, "B", "target"].values, [11, 12, 11, 12, 11, 12, 11])


def test_forecast_values_custom_weights(simple_df: TSDataset, naive_pipeline_1: Pipeline, naive_pipeline_2: Pipeline):
    """Check that VotingEnsemble gets average."""
    ensemble = VotingEnsemble(pipelines=[naive_pipeline_1, naive_pipeline_2], weights=[1, 3])
    ensemble.fit(ts=simple_df)
    forecast = ensemble.forecast()
    np.testing.assert_array_equal(forecast[:, "A", "target"].values, [47.25, 48, 47.25, 48, 47.25, 48, 47.25])
    np.testing.assert_array_equal(forecast[:, "B", "target"].values, [10.5, 12, 10.5, 12, 10.5, 12, 10.5])


def test_forecast_warning_prediction_intervals(
    simple_df: TSDataset, naive_pipeline_1: Pipeline, naive_pipeline_2: Pipeline
):
    """Check that VotingEnsemble warns when called with prediction intervals."""
    ensemble = VotingEnsemble(pipelines=[naive_pipeline_1, naive_pipeline_2], weights=[1, 3])
    ensemble.fit(ts=simple_df)
    with pytest.warns(UserWarning, match="doesn't support prediction intervals"):
        _ = ensemble.forecast(prediction_interval=True)


@pytest.mark.long
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


@pytest.mark.parametrize("n_jobs", (1, 5))
def test_backtest(voting_ensemble_pipeline: VotingEnsemble, example_tsds: TSDataset, n_jobs: int):
    """Check that backtest works with VotingEnsemble."""
    results = voting_ensemble_pipeline.backtest(ts=example_tsds, metrics=[MAE()], n_jobs=n_jobs, n_folds=3)
    for df in results:
        assert isinstance(df, pd.DataFrame)
