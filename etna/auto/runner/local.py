from typing import Callable
from typing import List
from typing import Optional
from typing import TypeVar

import dill
from joblib import Parallel
from joblib import delayed

from etna.auto.runner.base import AbstractRunner
from etna.auto.runner.utils import run_dill_encoded

T = TypeVar("T")


class LocalRunner(AbstractRunner):
    """LocalRunner for one threaded run."""

    def __call__(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Call given ``func`` with ``*args`` and ``**kwargs``."""
        return func(*args, **kwargs)


class ParallelLocalRunner(AbstractRunner):
    """ParallelLocalRunner for multiple parallel runs with joblib.

    Notes
    -----
    Global objects behavior could be different while parallel usage because platform dependent new process start.
    Be sure that new process is started with ``fork`` via ``multiprocessing.set_start_method``.
    If it's not possible you should try define all globals before ``if __name__ == "__main__"`` scope.
    """

    def __init__(
        self,
        n_jobs: int = 1,
        backend: str = "multiprocessing",
        mmap_mode: str = "c",
        joblib_params: Optional[dict] = None,
    ):
        """Init ParallelLocalRunner.

        Parameters
        ----------
        n_jobs:
            number of parallel jobs to use
        backend:
            joblib backend to use
        mmap_mode:
            joblib mmap mode
        joblib_params:
            joblib additional params
        """
        self.n_jobs = n_jobs
        self.backend = backend
        self.mmap_mode = mmap_mode
        self.joblib_params = {} if joblib_params is None else joblib_params

    def __call__(self, func: Callable[..., T], *args, **kwargs) -> List[T]:
        """Call given ``func`` with Joblib and ``*args`` and ``**kwargs``."""
        payload = dill.dumps((func, args, kwargs))
        job_results: List[T] = Parallel(
            n_jobs=self.n_jobs, backend=self.backend, mmap_mode=self.mmap_mode, **self.joblib_params
        )(delayed(run_dill_encoded)(payload) for _ in range(self.n_jobs))
        return job_results
