from etna.core import BaseMixin
from abc import ABC, abstractmethod
from typing import Optional
from scipy.sparse import csr_matrix
from etna.datasets import TSDataset


class Reconciliator(ABC, BaseMixin):
    """Base class to hold reconciliation methods."""
    def __init__(self, target_level: str, source_level: str):
        """Init Reconciliator.

        Parameters
        ----------
        target_level:
            Level to be reconciled from the forecasts.
        source_level:
            Level to be forecasted.
        """
        self.target_level = target_level
        self.source_level = source_level
        self.mapping_matrix: Optional[csr_matrix] = None

    @abstractmethod
    def fit(self, ts: TSDataset) -> "Reconciliator":
        """ Fit the reconciliator parameters.

        Parameters
        ----------
        ts:
            TSDataset on the level which is lower or equal to ``target_level``, ``source_level``.

        Returns
        -------
        :
            Fitted instance of reconciliator.
        """
        pass

    def aggregate(self, ts: TSDataset) -> TSDataset:
        """ Fit the reconciliator parameters.

        Parameters
        ----------
        ts:
            TSDataset on the level which is lower or equal to ``source_level``.

        Returns
        -------
        :
            TSDataset on the ``source_level``.
        """
        ts_aggregated = ts.get_level_dataset(target_level=self.source_level)
        return ts_aggregated
