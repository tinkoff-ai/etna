from typing import Dict
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from numpy.random import RandomState
from sklearn.ensemble import RandomForestRegressor

from etna.analysis import ModelRelevanceTable
from etna.analysis.feature_selection import mrmr
from etna.datasets import TSDataset
from etna.datasets.datasets_generation import generate_ar_df


@pytest.fixture
def df_with_regressors() -> Dict[str, pd.DataFrame]:
    num_segments = 3
    df = generate_ar_df(
        start_time="2020-01-01", periods=300, ar_coef=[1], sigma=1, n_segments=num_segments, random_seed=0, freq="D"
    )

    example_segment = df["segment"].unique()[0]
    timestamp = df[df["segment"] == example_segment]["timestamp"]
    df_exog = pd.DataFrame({"timestamp": timestamp})

    # useless regressors
    num_useless = 12
    df_regressors_useless = generate_ar_df(
        start_time="2020-01-01", periods=300, ar_coef=[1], sigma=1, n_segments=num_useless, random_seed=1, freq="D"
    )
    for i, segment in enumerate(df_regressors_useless["segment"].unique()):
        regressor = df_regressors_useless[df_regressors_useless["segment"] == segment]["target"].values
        df_exog[f"regressor_useless_{i}"] = regressor

    # useless categorical regressors
    num_cat_useless = 3
    for i in range(num_cat_useless):
        df_exog[f"categorical_regressor_useless_{i}"] = i
        df_exog[f"categorical_regressor_useless_{i}"] = df_exog[f"categorical_regressor_useless_{i}"].astype("category")

    # useful regressors: the same as target but with little noise
    df_regressors_useful = df.copy()
    sampler = RandomState(seed=2).normal
    for i, segment in enumerate(df_regressors_useful["segment"].unique()):
        regressor = df_regressors_useful[df_regressors_useful["segment"] == segment]["target"].values
        noise = sampler(scale=0.05, size=regressor.shape)
        df_exog[f"regressor_useful_{i}"] = regressor + noise

    # construct exog
    classic_exog_list = []
    for segment in df["segment"].unique():
        tmp = df_exog.copy(deep=True)
        tmp["segment"] = segment
        classic_exog_list.append(tmp)
    df_exog_all_segments = pd.concat(classic_exog_list)

    # construct TSDataset
    df = df[df["timestamp"] <= timestamp[200]]
    ts = TSDataset(df=TSDataset.to_dataset(df), df_exog=TSDataset.to_dataset(df_exog_all_segments), freq="D")
    return {
        "df": ts.to_pandas(),
        "target": TSDataset.to_dataset(df),
        "regressors": TSDataset.to_dataset(df_exog_all_segments),
    }


@pytest.mark.parametrize("fast_redundancy", [True, False])
@pytest.mark.parametrize(
    "relevance_method, expected_regressors",
    [(ModelRelevanceTable(), ["regressor_useful_0", "regressor_useful_1", "regressor_useful_2"])],
)
def test_mrmr_right_regressors(df_with_regressors, relevance_method, expected_regressors, fast_redundancy):
    relevance_table = relevance_method(
        df=df_with_regressors["target"], df_exog=df_with_regressors["regressors"], model=RandomForestRegressor()
    )
    selected_regressors = mrmr(
        relevance_table=relevance_table,
        regressors=df_with_regressors["regressors"],
        top_k=3,
        fast_redundancy=fast_redundancy,
    )
    assert set(selected_regressors) == set(expected_regressors)


@pytest.mark.parametrize("fast_redundancy", [True, False])
def test_mrmr_not_depend_on_columns_order(df_with_regressors, fast_redundancy):
    df, regressors = df_with_regressors["df"], df_with_regressors["regressors"]
    relevance_table = ModelRelevanceTable()(df=df, df_exog=regressors, model=RandomForestRegressor())
    expected_answer = mrmr(
        relevance_table=relevance_table, regressors=regressors, top_k=5, fast_redundancy=fast_redundancy
    )
    columns = list(regressors.columns.get_level_values("feature").unique())
    for i in range(10):
        np.random.shuffle(columns)
        answer = mrmr(
            relevance_table=relevance_table[columns],
            regressors=regressors.loc[pd.IndexSlice[:], pd.IndexSlice[:, columns]],
            top_k=5,
            fast_redundancy=fast_redundancy,
        )
        assert answer == expected_answer


@pytest.fixture()
def high_relevance_high_redundancy_problem(periods=10):
    relevance_table = pd.DataFrame(
        {"regressor_1": [1, 1], "regressor_2": [1, 1], "regressor_3": [1, 1]}, index=["segment_1", "segment_2"]
    )
    regressors = generate_ar_df(periods=periods, n_segments=2, start_time="2000-01-01", freq="D", random_seed=1).rename(
        columns={"target": "regressor_1"}
    )
    regressors["regressor_2"] = generate_ar_df(
        periods=periods, n_segments=2, start_time="2000-01-01", freq="D", random_seed=1
    )["target"]
    regressors["regressor_3"] = generate_ar_df(
        periods=periods, n_segments=2, start_time="2000-01-01", freq="D", random_seed=2
    )["target"]
    regressors = TSDataset.to_dataset(regressors)
    return {
        "relevance_table": relevance_table,
        "regressors": regressors,
        "expected_answer": ["regressor_1", "regressor_3"],
    }


@pytest.fixture()
def high_relevance_high_redundancy_problem_diff_starts(periods=10):
    relevance_table = pd.DataFrame(
        {"regressor_1": [1, 1], "regressor_2": [1, 1], "regressor_3": [1, 1]}, index=["segment_1", "segment_2"]
    )
    regressors = generate_ar_df(periods=periods, n_segments=2, start_time="2000-01-04", freq="D", random_seed=1).rename(
        columns={"target": "regressor_1"}
    )
    regressors["regressor_2"] = generate_ar_df(
        periods=periods, n_segments=2, start_time="2000-01-01", freq="D", random_seed=1
    )["target"]
    regressors["regressor_3"] = generate_ar_df(
        periods=periods, n_segments=2, start_time="2000-01-07", freq="D", random_seed=2
    )["target"]
    regressors = TSDataset.to_dataset(regressors)
    regressors.loc[pd.IndexSlice[:2], pd.IndexSlice[:, "regressor_1"]] = np.NaN
    regressors.loc[pd.IndexSlice[:4], pd.IndexSlice[:, "regressor_3"]] = np.NaN
    return {
        "relevance_table": relevance_table,
        "regressors": regressors,
        "expected_answer": ["regressor_1", "regressor_3"],
    }


@pytest.mark.parametrize("fast_redundancy", [True, False])
def test_mrmr_select_less_redundant_regressor(high_relevance_high_redundancy_problem, fast_redundancy):
    """Check that transform selects the less redundant regressor out of regressors with same relevance."""
    relevance_table, regressors = (
        high_relevance_high_redundancy_problem["relevance_table"],
        high_relevance_high_redundancy_problem["regressors"],
    )
    selected_regressors = mrmr(
        relevance_table=relevance_table, regressors=regressors, top_k=2, fast_redundancy=fast_redundancy
    )
    assert set(selected_regressors) == set(high_relevance_high_redundancy_problem["expected_answer"])


@pytest.mark.parametrize("fast_redundancy", [True, False])
def test_mrmr_select_less_redundant_regressor_diff_start(
    high_relevance_high_redundancy_problem_diff_starts, fast_redundancy
):
    """Check that transform selects the less redundant regressor out of regressors with same relevance."""
    relevance_table, regressors = (
        high_relevance_high_redundancy_problem_diff_starts["relevance_table"],
        high_relevance_high_redundancy_problem_diff_starts["regressors"],
    )
    selected_regressors = mrmr(
        relevance_table=relevance_table, regressors=regressors, top_k=2, fast_redundancy=fast_redundancy
    )
    assert set(selected_regressors) == set(high_relevance_high_redundancy_problem_diff_starts["expected_answer"])


def test_fast_redundancy_deprecation_warning(df_with_regressors):
    df, regressors = df_with_regressors["df"], df_with_regressors["regressors"]
    relevance_table = ModelRelevanceTable()(df=df, df_exog=regressors, model=RandomForestRegressor())
    with pytest.warns(DeprecationWarning, match="Option `fast_redundancy=False` was added for backward compatibility"):
        mrmr(relevance_table=relevance_table, regressors=regressors, top_k=2, fast_redundancy=False)


@pytest.mark.parametrize("fast_redundancy", [True, False])
def test_mrmr_with_castable_categorical_regressor(df_with_regressors, fast_redundancy):
    df, regressors = df_with_regressors["df"], df_with_regressors["regressors"]
    relevance_table = ModelRelevanceTable()(df=df, df_exog=regressors, model=RandomForestRegressor())
    mrmr(relevance_table=relevance_table, regressors=regressors, top_k=len(regressors), fast_redundancy=fast_redundancy)


@pytest.mark.parametrize("fast_redundancy", [True, False])
def test_mrmr_with_uncastable_categorical_regressor_fails(exog_and_target_dfs, fast_redundancy):
    df, regressors = exog_and_target_dfs
    with pytest.raises(ValueError, match="Only convertible to float features are allowed!"):
        mrmr(relevance_table=Mock(), regressors=regressors, top_k=len(regressors), fast_redundancy=fast_redundancy)
