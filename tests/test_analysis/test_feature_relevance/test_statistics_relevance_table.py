import numpy as np
import pandas as pd

from etna.analysis import get_statistics_relevance_table


def test_interface_statistics(simple_df_relevance):
    relevance_table = get_statistics_relevance_table(simple_df_relevance[0], simple_df_relevance[1])
    assert isinstance(relevance_table, pd.DataFrame)
    assert sorted(relevance_table.index) == sorted(
        simple_df_relevance[0].columns.get_level_values("segment").unique().tolist()
    )
    assert sorted(relevance_table.columns) == sorted(
        simple_df_relevance[1].columns.get_level_values("feature").unique().tolist()
    )


def test_statistics_relevance_table(simple_df_relevance):
    relevance_table = get_statistics_relevance_table(simple_df_relevance[0], simple_df_relevance[1])
    assert relevance_table["regressor_1"]["1"] < 1e-14
    assert relevance_table["regressor_1"]["2"] > 1e-1
    assert np.isnan(relevance_table["regressor_2"]["1"])
    assert relevance_table["regressor_2"]["2"] < 1e-10
