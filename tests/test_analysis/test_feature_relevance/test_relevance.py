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


def test_relevance_table_ranks(simple_df_relevance):
    rt = ModelRelevanceTable()
    df, df_exog = simple_df_relevance
    table = rt(df=df, df_exog=df_exog, return_ranks=True, model=DecisionTreeRegressor())
    assert table["regressor_1"]["1"] == 1
    assert table["regressor_2"]["1"] == 2
    assert table["regressor_1"]["2"] == 2
    assert table["regressor_2"]["2"] == 1
