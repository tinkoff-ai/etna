from dataclasses import asdict
from unittest.mock import Mock

import pytest

from etna.auto.utils import suggest_parameters
from etna.distributions import CategoricalDistribution
from etna.distributions import FloatDistribution
from etna.distributions import IntDistribution


@pytest.mark.parametrize(
    "params_to_tune",
    [
        {"x": CategoricalDistribution([False, True])},
        {"x": CategoricalDistribution(["A", "B", "C"])},
        {"x": IntDistribution(low=1, high=5)},
        {"x": IntDistribution(low=1, high=5, step=2)},
        {"x": IntDistribution(low=1, high=5, log=True)},
        {"x": FloatDistribution(low=0.1, high=10.0)},
        {"x": FloatDistribution(low=0.1, high=10.0, step=0.1)},
        {"x": FloatDistribution(low=1e-2, high=100, log=True)},
        {
            "c1": CategoricalDistribution([False, True]),
            "c2": CategoricalDistribution(["A", "B", "C"]),
            "i1": IntDistribution(low=1, high=5),
            "i2": IntDistribution(low=1, high=5, step=2),
            "f1": FloatDistribution(low=0.1, high=10.0),
            "f2": FloatDistribution(low=0.1, high=10.0, step=0.1),
        },
    ],
)
def test_suggest_parameters(params_to_tune):
    trial = Mock()
    method_names = {
        CategoricalDistribution: "suggest_categorical",
        IntDistribution: "suggest_int",
        FloatDistribution: "suggest_float",
    }

    results = suggest_parameters(trial=trial, params_to_tune=params_to_tune)

    assert results.keys() == params_to_tune.keys()
    for param_name in results.keys():
        method = getattr(trial, method_names[type(params_to_tune[param_name])])
        method.assert_any_call(param_name, **asdict(params_to_tune[param_name]))
