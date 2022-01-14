import numpy as np
import pandas as pd


def equals_with_nans(first_df: pd.DataFrame, second_df: pd.DataFrame) -> bool:
    """Compare two dataframes with consideration NaN == NaN is true."""
    if first_df.shape != second_df.shape:
        return False
    compare_result = (first_df.isna() & second_df.isna()) | (first_df == second_df)
    return np.all(compare_result)
