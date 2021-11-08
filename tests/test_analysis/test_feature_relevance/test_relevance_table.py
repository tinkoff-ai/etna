import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeRegressor

from etna.analysis.feature_relevance import get_model_relevance_table
from etna.analysis.feature_relevance import get_statistics_relevance_table


@pytest.mark.parametrize(
    "method,method_kwargs",
    ((get_statistics_relevance_table, {}), (get_model_relevance_table, {"model": DecisionTreeRegressor()})),
)
def test_interface(method, method_kwargs, simple_df_relevance):
    relevance_table = method(simple_df_relevance[0], simple_df_relevance[1], **method_kwargs)
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


def test_model_relevance_table(simple_df_relevance):
    relevance_table = get_model_relevance_table(
        simple_df_relevance[0], simple_df_relevance[1], model=DecisionTreeRegressor()
    )
    assert np.allclose(relevance_table["regressor_1"]["1"], 1)
    assert np.allclose(relevance_table["regressor_2"]["1"], 0)
    assert relevance_table["regressor_1"]["2"] < relevance_table["regressor_2"]["2"]
