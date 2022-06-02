import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeRegressor

from etna.analysis.feature_relevance import get_model_relevance_table
from etna.analysis.feature_relevance import get_statistics_relevance_table
from etna.datasets import TSDataset
from etna.datasets import duplicate_data


@pytest.mark.parametrize(
    "method,method_kwargs",
    ((get_statistics_relevance_table, {}), (get_model_relevance_table, {"model": DecisionTreeRegressor()})),
)
def test_interface(method, method_kwargs, simple_df_relevance):
    df, df_exog = simple_df_relevance
    relevance_table = method(df=df, df_exog=df_exog, **method_kwargs)
    assert isinstance(relevance_table, pd.DataFrame)
    assert sorted(relevance_table.index) == sorted(df.columns.get_level_values("segment").unique())
    assert sorted(relevance_table.columns) == sorted(df_exog.columns.get_level_values("feature").unique())


def test_statistics_relevance_table(simple_df_relevance):
    df, df_exog = simple_df_relevance
    relevance_table = get_statistics_relevance_table(df=df, df_exog=df_exog)
    assert relevance_table["regressor_1"]["1"] < 1e-14
    assert relevance_table["regressor_1"]["2"] > 1e-1
    assert np.isnan(relevance_table["regressor_2"]["1"])
    assert relevance_table["regressor_2"]["2"] < 1e-10


def test_model_relevance_table(simple_df_relevance):
    df, df_exog = simple_df_relevance
    relevance_table = get_model_relevance_table(df=df, df_exog=df_exog, model=DecisionTreeRegressor())
    assert np.allclose(relevance_table["regressor_1"]["1"], 1)
    assert np.allclose(relevance_table["regressor_2"]["1"], 0)
    assert relevance_table["regressor_1"]["2"] < relevance_table["regressor_2"]["2"]


@pytest.fixture()
def exog_and_target_dfs():
    seg = ["a"] * 30 + ["b"] * 30
    time = list(pd.date_range("2020-01-01", "2021-01-01")[:30])
    timestamps = time * 2
    target = np.arange(60)
    df = pd.DataFrame({"segment": seg, "timestamp": timestamps, "target": target})
    ts = TSDataset.to_dataset(df)

    cast = ["1.1"] * 10 + ["2"] * 9 + [None] + ["56.1"] * 10
    no_cast = ["1.1"] * 10 + ["two"] * 10 + ["56.1"] * 10
    none = [1] * 10 + [2] * 10 + [56.1] * 10
    none[10] = None
    df = pd.DataFrame(
        {
            "timestamp": time,
            "exog1": np.arange(100, 70, -1),
            "exog2": np.sin(np.arange(30) / 10),
            "exog3": np.exp(np.arange(30)),
            "cast": cast,
            "no_cast": no_cast,
            "none": none,
        }
    )
    df["cast"] = df["cast"].astype("category")
    df["no_cast"] = df["no_cast"].astype("category")
    df_exog = duplicate_data(df, segments=["a", "b"])
    return ts, df_exog


@pytest.fixture()
def exog_and_target_dfs_with_none():
    seg = ["a"] * 30 + ["b"] * 30
    time = list(pd.date_range("2020-01-01", "2021-01-01")[:30])
    timestamps = time * 2
    target = np.arange(60, dtype=float)
    target[5] = np.nan
    df = pd.DataFrame({"segment": seg, "timestamp": timestamps, "target": target})
    ts = TSDataset.to_dataset(df)

    none = [1] * 10 + [2] * 10 + [56.1] * 10
    none[10] = None
    df = pd.DataFrame(
        {
            "timestamp": time,
            "exog1": np.arange(100, 70, -1),
            "exog2": np.sin(np.arange(30) / 10),
            "exog3": np.exp(np.arange(30)),
            "none": none,
        }
    )
    df_exog = duplicate_data(df, segments=["a", "b"])
    return ts, df_exog


@pytest.mark.parametrize(
    "columns,match",
    (
        (["exog1", "exog2", "exog3", "cast"], "Exogenous data contains columns with category type"),
        (["exog1", "exog2", "exog3", "none"], "Exogenous or target data contains None"),
    ),
)
def test_warnings_statistic_table(columns, match, exog_and_target_dfs):
    df, df_exog = exog_and_target_dfs
    df_exog = df_exog[[i for i in df_exog.columns if i[1] in columns]]
    with pytest.warns(UserWarning, match=match):
        get_statistics_relevance_table(df=df, df_exog=df_exog)


def test_errors_statistic_table(exog_and_target_dfs):
    df, df_exog = exog_and_target_dfs
    with pytest.raises(ValueError, match="column cannot be cast to float type!"):
        get_statistics_relevance_table(df=df, df_exog=df_exog)


def test_work_statistic_table(exog_and_target_dfs):
    df, df_exog = exog_and_target_dfs
    df_exog = df_exog[[i for i in df_exog.columns if i[1] != "no_cast"]]
    get_statistics_relevance_table(df=df, df_exog=df_exog)


def test_target_none_statistic_table(exog_and_target_dfs_with_none):
    df, df_exog = exog_and_target_dfs_with_none
    df_exog = df_exog[[i for i in df_exog.columns if i[1][:-1] == "exog"]]
    with pytest.warns(UserWarning, match="Exogenous or target data contains None"):
        get_statistics_relevance_table(df=df, df_exog=df_exog)


def test_target_none_model_table(exog_and_target_dfs_with_none):
    df, df_exog = exog_and_target_dfs_with_none
    df_exog = df_exog[[i for i in df_exog.columns if i[1][:-1] == "exog"]]
    with pytest.warns(UserWarning, match="Exogenous or target data contains None"):
        get_model_relevance_table(df=df, df_exog=df_exog, model=DecisionTreeRegressor())


def test_exog_none_model_table(exog_and_target_dfs):
    df, df_exog = exog_and_target_dfs
    df_exog = df_exog[[i for i in df_exog.columns if i[1] in ["exog1", "exog2", "exog3", "none"]]]
    with pytest.warns(UserWarning, match="Exogenous or target data contains None"):
        get_model_relevance_table(df=df, df_exog=df_exog, model=DecisionTreeRegressor())


def test_exog_and_target_none_statistic_table(exog_and_target_dfs_with_none):
    df, df_exog = exog_and_target_dfs_with_none
    with pytest.warns(UserWarning, match="Exogenous or target data contains None"):
        get_statistics_relevance_table(df=df, df_exog=df_exog)


def test_exog_and_target_none_model_table(exog_and_target_dfs_with_none):
    df, df_exog = exog_and_target_dfs_with_none
    with pytest.warns(UserWarning, match="Exogenous or target data contains None"):
        get_model_relevance_table(df=df, df_exog=df_exog, model=DecisionTreeRegressor())
