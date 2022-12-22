from etna.core import BaseMixin
from abc import ABC
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
