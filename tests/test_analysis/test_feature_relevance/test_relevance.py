from etna.analysis.feature_relevance import StatisticsRelevanceTable


def test_statistics_relevance_table(simple_df_relevance):
    rt = StatisticsRelevanceTable()
    assert not rt.greater_is_better
    df, df_exog = simple_df_relevance
    assert rt(df=df, df_exog=df_exog).shape == (2, 2)
