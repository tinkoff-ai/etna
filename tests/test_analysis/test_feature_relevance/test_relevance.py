import pytest
from sklearn.tree import DecisionTreeRegressor

from etna.analysis.feature_relevance import ModelRelevanceTable
from etna.analysis.feature_relevance import StatisticsRelevanceTable


def test_statistics_relevance_table(simple_df_relevance):
    rt = StatisticsRelevanceTable()
    assert not rt.greater_is_better
    df, df_exog = simple_df_relevance
    assert rt(df=df, df_exog=df_exog, return_ranks=False).shape == (2, 2)


def test_model_relevance_table(simple_df_relevance):
    rt = ModelRelevanceTable()
    assert rt.greater_is_better
    df, df_exog = simple_df_relevance
    assert rt(df=df, df_exog=df_exog, return_ranks=False, model=DecisionTreeRegressor()).shape == (2, 2)


@pytest.mark.parametrize(
    "greater_is_better,answer",
    ((True, [1, 2, 2, 1]), (False, [2, 1, 1, 2])),
)
def test_relevance_table_ranks(greater_is_better, answer, simple_df_relevance):
    rt = ModelRelevanceTable()
    rt.greater_is_better = greater_is_better
    df, df_exog = simple_df_relevance
    table = rt(df=df, df_exog=df_exog, return_ranks=True, model=DecisionTreeRegressor())
    assert table["regressor_1"]["1"] == answer[0]
    assert table["regressor_2"]["1"] == answer[1]
    assert table["regressor_1"]["2"] == answer[2]
    assert table["regressor_2"]["2"] == answer[3]
