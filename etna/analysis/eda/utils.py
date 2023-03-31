from typing import TYPE_CHECKING
from typing import List
from typing import Optional

import numpy as np

if TYPE_CHECKING:
    from etna.datasets import TSDataset


def get_correlation_matrix(
    ts: "TSDataset",
    columns: Optional[List[str]] = None,
    segments: Optional[List[str]] = None,
    method: str = "pearson",
) -> np.ndarray:
    """Compute pairwise correlation of timeseries for selected segments.

    Parameters
    ----------
    ts:
        TSDataset with timeseries data
    columns:
        Columns to use, if None use all columns
    segments:
        Segments to use
    method:
        Method of correlation:

        * pearson: standard correlation coefficient

        * kendall: Kendall Tau correlation coefficient

        * spearman: Spearman rank correlation

    Returns
    -------
    np.ndarray
        Correlation matrix
    """
    if method not in ["pearson", "kendall", "spearman"]:
        raise ValueError(f"'{method}' is not a valid method of correlation.")

    if segments is None:
        segments = sorted(ts.segments)
    if columns is None:
        columns = list(set(ts.df.columns.get_level_values("feature")))

    correlation_matrix = ts[:, segments, columns].corr(method=method).values
    return correlation_matrix
