from abc import ABC
from abc import abstractmethod
from typing import Optional

import pandas as pd
import scipy
from scipy.sparse import csr_matrix

from etna.core import BaseMixin
from etna.datasets import TSDataset


class BaseReconciliator(ABC, BaseMixin):
    """Base class to hold reconciliation methods."""

    def __init__(self, target_level: str, source_level: str):
        """Init BaseReconciliator.

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
    def fit(self, ts: TSDataset) -> "BaseReconciliator":
        """Fit the reconciliator parameters.

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
        """Aggregate the dataset to the ``source_level``.

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

    def reconcile(self, ts: TSDataset) -> TSDataset:
        """Reconcile the forecasts in the dataset.

        Parameters
        ----------
        ts:
            TSDataset on the ``source_level``.

        Returns
        -------
        :
            TSDataset on the ``target_level``.
        """
        if ts.current_df_level != self.source_level:
            raise ValueError(f"Dataset should be on the {self.source_level} level!")

        current_level_segments = ts.hierarchical_structure.get_level_segments(level_name=self.source_level)
        target_level_segments = ts.hierarchical_structure.get_level_segments(level_name=self.target_level)

        current_level_values = ts[:, current_level_segments, "target"].values.T
        target_level_values = scipy.sparse.csr_matrix.dot(self.mapping_matrix, current_level_values)

        df_reconciled = pd.DataFrame(
            target_level_values, columns=pd.MultiIndex.from_product([target_level_segments, ["target"]]), index=ts.index
        )
        ts_reconciled = TSDataset(
            df=df_reconciled,
            freq=ts.freq,
            df_exog=ts.df_exog,
            known_future=ts.known_future,
            hierarchical_structure=ts.hierarchical_structure,
        )
        return ts_reconciled
