from typing import Callable
from typing import Optional
from typing import Sequence
from typing import Union

import optuna
from optuna.pruners import BasePruner
from optuna.samplers import BaseSampler
from optuna.storages import BaseStorage
from optuna.study import Study
from optuna.study import StudyDirection
from optuna.trial import Trial
from typing_extensions import Literal

from etna.auto.runner import AbstractRunner
from etna.auto.runner import LocalRunner

OptunaDirection = Literal["minimize", "maximize"]


class Optuna:
    """Class for encapsulate work with Optuna."""

    def __init__(
        self,
        direction: Union[OptunaDirection, StudyDirection],
        study_name: Optional[str] = None,
        sampler: Optional[BaseSampler] = None,
        storage: Optional[BaseStorage] = None,
        pruner: Optional[BasePruner] = None,
        directions: Optional[Sequence[Union[OptunaDirection, StudyDirection]]] = None,
        load_if_exists: bool = True,
    ):
        """Init wrapper for Optuna.

        Parameters
        ----------
        direction:
            optuna direction
        study_name:
            name of study
        sampler:
            optuna sampler to use
        storage:
            storage to use
        pruner:
            optuna pruner
        directions:
            directions to optimize in case of multi-objective optimization
        load_if_exists:
            load study from storage if it exists or raise exception if it doesn't
        """
        self._study = optuna.create_study(
            storage=storage,
            study_name=study_name,
            direction=direction,
            sampler=sampler,
            load_if_exists=load_if_exists,
            pruner=pruner,
            directions=directions,
        )

    def tune(
        self,
        objective: Callable[[Trial], Union[float, Sequence[float]]],
        n_trials: Optional[int] = None,
        timeout: Optional[int] = None,
        runner: Optional[AbstractRunner] = None,
        **kwargs,
    ):
        """Call optuna ``optimize`` for chosen Runner.

        Parameters
        ----------
        objective:
            objective function to optimize in optuna style
        n_trials:
            number of trials to run. N.B. in case of parallel runner, this is number of trials per worker
        timeout:
            timeout for optimization. N.B. in case of parallel runner, this is timeout per worker
        kwargs:
            additional arguments to pass to :py:meth:`optuna.study.Study.optimize`
        """
        if runner is None:
            runner = LocalRunner()
        _ = runner(self.study.optimize, objective, n_trials=n_trials, timeout=timeout, **kwargs)

    @property
    def study(self) -> Study:
        """Get optuna study."""
        return self._study
