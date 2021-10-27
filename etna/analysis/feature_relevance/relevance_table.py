from typing import Dict

import pandas as pd

try:
    import sys
    from pathlib import Path

    tsfresh_path = Path(__file__).parents[2] / "libs" / "tsfresh"
    sys.path.append(str(tsfresh_path))
    from tsfresh.feature_selection.relevance import calculate_relevance_table
except Exception as e:
    raise e
finally:
    sys.path.remove(str(tsfresh_path))


def get_statistics_relevance_table(df: pd.DataFrame, df_exog: pd.DataFrame) -> pd.DataFrame:
    """Calculate relevance table with p-values from tsfresh.

    Parameters
    ----------
    df:
        dataframe with timeseries
    df_exog:
        dataframe with exogenous data

    Returns
    -------
    dataframe with p-values.
    """
    regressors = df_exog.columns.get_level_values("feature").unique().tolist()
    result: Dict[str, list] = dict(zip(regressors, [[] for i in range(len(regressors))]))
    result_segment = []
    for seg in df.columns.get_level_values("segment").unique().tolist():
        result_segment.append(seg)
        first_valid_idx = df.loc[:, seg].first_valid_index()
        df_now = df.loc[first_valid_idx:, seg]["target"]
        df_exog_now = df_exog.loc[:, seg][first_valid_idx:]
        relevance = calculate_relevance_table(df_exog_now[: len(df_now)], df_now)[["feature", "p_value"]].values
        for regr, value in relevance:
            result[regr].append(value)
    relevance_table = pd.DataFrame(result)
    relevance_table.index = result_segment
    return relevance_table
