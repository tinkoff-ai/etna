from typing import List
from typing import Optional
from typing import Set

import numpy as np
from optuna.samplers import BaseSampler
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.trial import TrialState

from etna.auto.utils import config_hash
from etna.auto.utils import retry


class ConfigSampler(BaseSampler):
    """Optuna based sampler for greedy search over different configurations."""

    def __init__(self, configs: List[dict], random_generator: Optional[np.random.Generator] = None, retries: int = 10):
        """Init Config sampler.

        Parameters
        ----------
        configs:
            pool of configs to sample from
        random_generator:
            numpy generator to get reproducible samples
        retries:
            number of retries to get new sample from storage. It could be useful if storage is not reliable.
        """
        self.configs = configs
        self.configs_hash = {config_hash(config=config): config for config in self.configs}
        self._rng = random_generator
        self.retries = retries

    def sample_independent(self, *args, **kwargs):  # noqa: D102
        """Sample independent. Not used."""
        return {}

    def infer_relative_search_space(self, *args, **kwargs):  # noqa: D102
        """Infer relative search space. Not used."""
        return {}

    def sample_relative(self, study: Study, trial: FrozenTrial, *args, **kwargs) -> dict:
        """Sample configuration to test.

        Parameters
        ----------
        study:
            current optuna study
        trial:
            optuna trial to use

        Return
        ------
        :
            sampled configuration to run objective on
        """
        trials_to_sample = self._get_unfinished_hashes(study=study, current_trial=trial)

        if len(trials_to_sample) == 0:
            # TODO: this could cause job duplication
            # For some reason `_get_unfinished_hashes` does not return zero length list in `after_trial`
            _to_sample = list(self.configs_hash)
            idx = self.rng.choice(len(_to_sample))
            hash_to_sample = _to_sample[idx]
        else:
            _trials_to_sample = list(trials_to_sample)
            idx = self.rng.choice(len(_trials_to_sample))
            hash_to_sample = _trials_to_sample[idx]

        map_to_objective = self.configs_hash[hash_to_sample]

        study._storage.set_trial_user_attr(trial._trial_id, "hash", hash_to_sample)
        study._storage.set_trial_user_attr(trial._trial_id, "pipeline", map_to_objective)
        return map_to_objective

    def after_trial(self, study: Study, trial: FrozenTrial, *args, **kwargs) -> None:  # noqa: D102
        """Stop study if all configs have been tested.

        Parameters
        ----------
        study:
            current optuna study
        """
        unfinished_hashes = self._get_unfinished_hashes(study=study, current_trial=trial)

        if len(unfinished_hashes) == 0:
            study.stop()
        if len(unfinished_hashes) == 1 and list(unfinished_hashes)[0] == trial.user_attrs["hash"]:
            study.stop()

    def _get_unfinished_hashes(self, study: Study, current_trial: Optional[FrozenTrial] = None) -> Set[str]:
        """Get unfinished config hashes.

        Parameters
        ----------
        study:
            current optuna study

        Returns
        -------
        :
            hashes to run
        """
        trials = study._storage.get_all_trials(study._study_id, deepcopy=False)

        if current_trial is not None:
            trials = [trial for trial in trials if trial._trial_id != current_trial._trial_id]

        finished_trials_hash = []
        running_trials_hash = []

        for t in trials:
            if t.state.is_finished():
                finished_trials_hash.append(t.user_attrs["hash"])
            elif t.state == TrialState.RUNNING:

                def _closure():
                    return study._storage.get_trial(t._trial_id).user_attrs["hash"]

                hash_to_add = retry(_closure, max_retries=self.retries)
                running_trials_hash.append(hash_to_add)
            else:
                pass

        return set(self.configs_hash) - set(finished_trials_hash) - set(running_trials_hash)

    @property
    def rng(self):  # noqa: D102
        if self._rng is None:
            self._rng = np.random.default_rng()
        return self._rng

    def get_config_by_hash(self, hash: str):
        """Get config by hash.

        Parameters
        ----------
        hash:
            hash to get config for
        """
        return self.configs_hash[hash]
