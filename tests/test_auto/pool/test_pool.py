from copy import deepcopy

import pytest

from etna.auto.pool import Pool
from etna.auto.pool.templates import DEFAULT
from etna.datasets import TSDataset
from etna.pipeline import Pipeline


def test_generate_config():
    pipelines = Pool.default.value.generate(horizon=1)
    assert len(pipelines) == len(DEFAULT)


@pytest.mark.long_2
def test_default_pool_fit_predict(example_reg_tsds):
    horizon = 7
    pipelines = Pool.default.value.generate(horizon=horizon)

    def fit_predict(pipeline: Pipeline) -> TSDataset:
        pipeline.fit(deepcopy(example_reg_tsds))
        ts_forecast = pipeline.forecast()
        return ts_forecast

    ts_forecasts = [fit_predict(pipeline) for pipeline in pipelines]

    for ts_forecast in ts_forecasts:
        assert len(ts_forecast.to_pandas()) == horizon
