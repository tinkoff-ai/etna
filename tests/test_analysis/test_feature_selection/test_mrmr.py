from typing import List
from typing import Tuple

import numpy as np
import pandas as pd
import pytest
from numpy.random import RandomState
from sklearn.datasets import make_classification

from etna.analysis import StatisticsRelevanceTable
from etna.analysis.feature_selection import mrmr
from etna.clustering import EuclideanClustering
from etna.datasets import TSDataset
from etna.datasets.datasets_generation import generate_ar_df
from etna.transforms import MRMRFeatureSelectionTransform


@pytest.fixture
def df_with_regressors() -> pd.DataFrame:
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
    return ts.to_pandas()


@pytest.fixture()
def random_classification_task(random_seed) -> Tuple[pd.DataFrame, np.ndarray]:
    x, y = make_classification(
        n_features=30, n_informative=4, n_redundant=0, n_repeated=0, shuffle=False, random_state=random_seed
    )
    x = pd.DataFrame(x)
    return x, y


@pytest.fixture()
def random_classification_task_with_nans(random_seed) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    x, y = make_classification(
        n_features=30, n_informative=4, n_redundant=0, n_repeated=0, shuffle=False, random_state=random_seed
    )
    x = pd.DataFrame(x)
    x.iloc[:, [-1, -2]] = np.NAN
    x.iloc[0, -3] = np.NAN
    return x, y, list(x.columns[:-2])


@pytest.mark.parametrize(
    "relevance_method, clustering_method, expected_regressors",
    [
        (
            StatisticsRelevanceTable(),
            EuclideanClustering(),
            ["regressor_useful_1", "regressor_useful_2", "regressor_useless_9"],
        ),
    ],
)
def test_mrmr_right_regressors(df_with_regressors, relevance_method, clustering_method, expected_regressors):
    """Check that transform selects right top_k regressors."""
    mrmr = MRMRFeatureSelectionTransform(
        relevance_method=relevance_method, top_k=3, clustering_method=clustering_method, n_clusters=2
    )
    mrmr.fit(df_with_regressors)
    assert set(mrmr.selected_regressors) == set(expected_regressors)


def test_mrmr_not_depend_on_columns_order(random_classification_task):
    x, y = random_classification_task
    expected_answer = mrmr(x=x, y=y, top_k=5)
    columns = list(x.columns)
    for i in range(10):
        np.random.shuffle(columns)
        answer = mrmr(x=x[columns], y=y, top_k=5)
        assert answer == expected_answer


def test_mrmr_not_depend_on_rows_order(random_classification_task):
    x, y = random_classification_task
    expected_answer = mrmr(x=x, y=y, top_k=5)
    index = list(X.index)
    for i in range(10):
        np.random.shuffle(index)
        answer = mrmr(x=x.iloc[index], y=y, top_k=5)
        assert answer == expected_answer


def test_mrmr_work_with_nans(random_classification_task_with_nans):
    x, y, expected_answer = random_classification_task_with_nans
    answer = mrmr(x=x, y=y, top_k=x.shape[1])
    assert sorted(answer) == sorted(expected_answer)
