from functools import partial

import numpy as np
import pytest

from etna.auto.runner import LocalRunner
from etna.auto.runner import ParallelLocalRunner


@pytest.fixture()
def payload():
    func = partial(np.einsum, "ni,im->nm")
    args = np.random.normal(size=(10, 20)), np.random.normal(size=(20, 5))
    return func, args


def test_run_local_runner(payload):
    func, args = payload
    runner = LocalRunner()
    result = runner(func, *args)
    assert result.shape == (args[0].shape[0], args[1].shape[1])


def test_run_parallel_local_runner(payload):
    func, args = payload
    n_jobs = 4
    runner = ParallelLocalRunner(n_jobs=n_jobs)
    result = runner(func, *args)
    assert len(result) == n_jobs
    for res in result:
        assert res.shape == (args[0].shape[0], args[1].shape[1])
