from typing import Tuple

import numpy as np
import pandas as pd


def _get_fictitious_relevances(pvalues: pd.DataFrame, alpha: float) -> Tuple[np.ndarray, float]:
    """
    Convert p-values into fictitious variables, with function f(x) = 1 - x.

    Also converts alpha into fictitious variable.

    Parameters
    ----------
    pvalues:
        dataFrame with pvalues
    alpha:
        significance level, default alpha = 0.05

    Returns
    -------
    pvalues:
        array with fictitious relevances
    new_alpha:
        adjusted significance level
    """
    pvalues = 1 - pvalues
    new_alpha = 1 - alpha
    return pvalues, new_alpha
