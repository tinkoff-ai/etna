from etna.datasets import TSDataset
from etna.reconciliation.base import BaseReconciliator


class BottomUpReconciliator(BaseReconciliator):
    """Bottom-up reconciliation."""

    def __init__(self, target_level: str, source_level: str):
        """Create bottom-up reconciliator from ``source_level`` to ``target_level``.

        Parameters
        ----------
        target_level:
            Level to be reconciled from the forecasts.
        source_level:
            Level to be forecasted.
        """
        super().__init__(target_level=target_level, source_level=source_level)

    def fit(self, ts: TSDataset) -> "BottomUpReconciliator":
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

        if source_level_index < target_level_index:
            raise ValueError("Source level should be lower or equal in the hierarchy than the target level!")

        if current_level_index < source_level_index:
            raise ValueError("Current TSDataset level should be lower or equal in the hierarchy than the source level!")

        self.mapping_matrix = ts.hierarchical_structure.get_summing_matrix(
            target_level=self.target_level, source_level=self.source_level
        )

        return self
