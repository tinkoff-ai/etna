from abc import ABC
from abc import abstractmethod
from typing import Dict

import pandas as pd

from etna.core import BaseMixin


class Clustering(ABC, BaseMixin):
    """Base class for ETNA clustering algorithms."""

    @abstractmethod
    def fit_predict(self) -> Dict[str, int]:
        """Fit clustering algo and predict clusters.

        Returns
        -------
        Dict[str, int]:
            dict in format {segment: cluster}
        """
        pass

    @abstractmethod
    def get_centroids(self) -> pd.DataFrame:
        """Get centroids of clusters.

        Returns
        -------
        pd.DataFrame:
            dataframe with centroids
        """
        pass
