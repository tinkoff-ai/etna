from abc import ABC
from abc import abstractmethod
from typing import Dict

import pandas as pd

from etna.core import BaseMixin


class Clustering(ABC, BaseMixin):
    """Base class for ETNA clustering algorithms."""

    def __init__(self, n_jobs: int = 1):
        """Init Clustering.

        Parameters
        ----------
        n_jobs:
            number of jobs to run in parallel
        """
        self.n_jobs = n_jobs

    @abstractmethod
    def fit_predict(self) -> Dict[str, int]:
        """Fit clustering algo and predict clusters.

        Returns
        -------
        segment-cluster dict:
            dict in format {segment: cluster}
        """
        pass

    @abstractmethod
    def get_centroids(self) -> pd.DataFrame:
        """Get centroids of clusters.

        Returns
        -------
        centroids_df:
            dataframe with centroids
        """
        pass
