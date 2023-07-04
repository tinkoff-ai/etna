import json
import random
import time
from dataclasses import asdict
from hashlib import md5
from typing import Any
from typing import Callable
from typing import Dict

import optuna

from etna.distributions import BaseDistribution
from etna.distributions import CategoricalDistribution
from etna.distributions import FloatDistribution
from etna.distributions import IntDistribution


def config_hash(config: dict):
    """Compute hash of given ``config``."""
    return md5(json.dumps(config, sort_keys=True).encode()).hexdigest()


def retry(func: Callable[..., Any], max_retries: int = 5, sleep_time: int = 10, jitter: int = 10) -> Any:
    """Retry function call with jitter."""
    rng = random.SystemRandom()
    for i in range(max_retries + 1):
        try:
            value = func()
            return value
        except Exception as e:
            if i < max_retries:
                time.sleep(rng.random() * jitter + sleep_time)
                continue
            else:
                raise e


def suggest_parameters(trial: optuna.Trial, params_to_tune: Dict[str, BaseDistribution]) -> Dict[str, Any]:
    """Suggest parameters for a trial."""
    distributions = {
        CategoricalDistribution: "suggest_categorical",
        IntDistribution: "suggest_int",
        FloatDistribution: "suggest_float",
    }

    params_suggested = {}
    for param_name, param_distr in params_to_tune.items():
        method_name = distributions[type(param_distr)]
        method_kwargs = asdict(param_distr)
        method = getattr(trial, method_name)
        params_suggested[param_name] = method(param_name, **method_kwargs)

    return params_suggested
