from abc import ABC
from abc import abstractmethod
from typing import Optional

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from etna.core import BaseMixin
from etna.datasets import TSDataset
from etna.transforms.utils import match_target_quantiles


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
        if self.mapping_matrix is None:
            raise ValueError(f"Reconciliator is not fitted!")

        if ts.hierarchical_structure is None:
            raise ValueError(f"Passed dataset has no hierarchical structure!")

        if ts.current_df_level != self.source_level:
            raise ValueError(f"Dataset should be on the {self.source_level} level!")

        current_level_segments = ts.hierarchical_structure.get_level_segments(level_name=self.source_level)
        target_level_segments = ts.hierarchical_structure.get_level_segments(level_name=self.target_level)

        column_names = ts.columns.get_level_values(level=1)
        target_names = tuple(match_target_quantiles(column_names))
        target_names = target_names + ("target",)

        num_target_names = len(target_names)
        num_current_level_segments = len(current_level_segments)
        num_target_level_segments = len(target_level_segments)

        current_level_values = ts[:, current_level_segments, target_names].values

        current_level_values = current_level_values.reshape((-1, num_current_level_segments, num_target_names))
        current_level_values = np.swapaxes(current_level_values, 1, 2)
        current_level_values = current_level_values.reshape((-1, num_current_level_segments))

        target_level_values = current_level_values @ self.mapping_matrix.T

        target_level_values = target_level_values.reshape((-1, num_target_names, num_target_level_segments))
        target_level_values = np.swapaxes(target_level_values, 1, 2)
        target_level_values = target_level_values.reshape((-1, num_target_names * num_target_level_segments))

        df_reconciled = pd.DataFrame(
            target_level_values,
            columns=pd.MultiIndex.from_product([target_level_segments, target_names], names=["segment", "feature"]),
            index=ts.index,
        )
        ts_reconciled = TSDataset(
            df=df_reconciled,
            freq=ts.freq,
            df_exog=ts.df_exog,
            known_future=ts.known_future,
            hierarchical_structure=ts.hierarchical_structure,
        )
        return ts_reconciled
