import pytest
from optuna.pruners import MedianPruner
from optuna.samplers import GridSampler
from optuna.samplers import TPESampler
from optuna.storages import InMemoryStorage
from optuna.study import StudyDirection

from etna.auto.optuna import Optuna


@pytest.fixture()
def grid_sampler():
    return GridSampler({"x": [-2, -1, 0, 1, 2], "y": [1, 0, 1]})


@pytest.fixture()
def objective():
    def _objective(trial):
        x = trial.suggest_uniform("x", -2, 2)
        y = trial.suggest_uniform("y", -1, 1)
        return x**2 + y**2

    return _objective


def test_optuna_with_grid(grid_sampler, objective, expected_best_params={"x": 0, "y": 0}):
    opt = Optuna("minimize", sampler=grid_sampler)
    opt.tune(objective, gc_after_trial=True)
    assert opt.study.best_params == expected_best_params


def test_optuna_init():
    opt = Optuna("maximize")
    assert isinstance(opt.study.sampler, TPESampler)
    assert opt.study.direction == StudyDirection.MAXIMIZE
    assert isinstance(opt.study.pruner, MedianPruner)
    assert isinstance(opt.study._storage, InMemoryStorage)
