from enum import Enum

import bottleneck as bn
import pandas as pd
from scipy.sparse import lil_matrix

from etna.datasets import TSDataset
from etna.reconciliation.base import BaseReconciliator


class ReconciliationProportionsMethod(str, Enum):
    """Enum for different reconciliation proportions methods."""

    AHP = "AHP"
    PHA = "PHA"

    @classmethod
    def _missing_(cls, method):
        raise ValueError(
            f"Unable to recognize reconciliation method '{method}'! "
            f"Supported methods: {', '.join(sorted(m for m in cls))}."
        )


class TopDownReconciliator(BaseReconciliator):
    """Top-down reconciliation methods.

    Notes
    -----
    Top-down reconciliation methods support only non-negative data.
    """

    def __init__(self, target_level: str, source_level: str, period: int, method: str):
        """Create top-down reconciliator from ``source_level`` to ``target_level``.

        Parameters
        ----------
        target_level:
            Level to be reconciled from the forecasts.
        source_level:
            Level to be forecasted.
        period:
            Period length for calculation reconciliation proportions.
        method:
            Proportions calculation method. Selects last ``period`` timestamps for estimation.
            Currently supported options:

            * AHP - Average historical proportions

            * PHA - Proportions of the historical averages
        """
        super().__init__(target_level=target_level, source_level=source_level)

        if period < 1:
            raise ValueError("Period length must be positive!")

        self.period = period
        self.method = method

        proportions_method = ReconciliationProportionsMethod(method)
        if proportions_method == ReconciliationProportionsMethod.AHP:
            self._proportions_method_func = self._estimate_ahp_proportion
        elif proportions_method == ReconciliationProportionsMethod.PHA:
            self._proportions_method_func = self._estimate_pha_proportion
        else:
            raise ValueError(f"Failed to initialize proportions calculation method with name '{method}'!")

    def fit(self, ts: TSDataset) -> "TopDownReconciliator":
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
        if ts.hierarchical_structure is None:
            raise ValueError(f"The method can be applied only to instances with a hierarchy!")

        current_level_index = ts.hierarchical_structure.get_level_depth(ts.current_df_level)  # type: ignore
        source_level_index = ts.hierarchical_structure.get_level_depth(self.source_level)
        target_level_index = ts.hierarchical_structure.get_level_depth(self.target_level)

        if target_level_index < source_level_index:
            raise ValueError("Target level should be lower or equal in the hierarchy than the source level!")

        if current_level_index < target_level_index:
            raise ValueError("Current TSDataset level should be lower or equal in the hierarchy than the target level!")

        if (ts[..., "target"] < 0).values.any():
            raise ValueError("Provided dataset should not contain any negative numbers!")

        source_level_ts = ts.get_level_dataset(self.source_level)
        target_level_ts = ts.get_level_dataset(self.target_level)

        if source_level_index < target_level_index:

            summing_matrix = target_level_ts.hierarchical_structure.get_summing_matrix(  # type: ignore
                target_level=self.source_level, source_level=self.target_level
            )

            source_level_segments = source_level_ts.hierarchical_structure.get_level_segments(self.source_level)  # type: ignore
            target_level_segments = target_level_ts.hierarchical_structure.get_level_segments(self.target_level)  # type: ignore

            self.mapping_matrix = lil_matrix((len(target_level_segments), len(source_level_segments)))

            for source_index, target_index in zip(*summing_matrix.nonzero()):
                source_segment = source_level_segments[source_index]
                target_segment = target_level_segments[target_index]

                self.mapping_matrix[target_index, source_index] = self._proportions_method_func(  # type: ignore
                    target_series=target_level_ts[:, target_segment, "target"],
                    source_series=source_level_ts[:, source_segment, "target"],
                )

            self.mapping_matrix = self.mapping_matrix.tocsr()

        else:
            self.mapping_matrix = target_level_ts.hierarchical_structure.get_summing_matrix(  # type: ignore
                target_level=self.target_level, source_level=self.source_level
            )

        return self

    def _estimate_ahp_proportion(self, target_series: pd.Series, source_series: pd.Series) -> float:
        """Calculate reconciliation proportion with Average historical proportions method."""
        data = pd.concat((target_series, source_series), axis=1).values
        data = data[-self.period :]
        return bn.nanmean(data[..., 0] / data[..., 1])

    def _estimate_pha_proportion(self, target_series: pd.Series, source_series: pd.Series) -> float:
        """Calculate reconciliation proportion with Proportions of the historical averages method."""
        target_data = target_series.values
        source_data = source_series.values
        return bn.nanmean(target_data[-self.period :]) / bn.nanmean(source_data[-self.period :])
