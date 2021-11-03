from typing import Dict, List

from etna.transforms.gale_shapley import GaleShapleyFeatureSelectionTransform
import pytest
from etna.datasets import generate_ar_df, TSDataset
import pandas as pd


@pytest.fixture
def ts_with_complex_exog() -> TSDataset:
    df = generate_ar_df(periods=100, start_time="2020-01-01", n_segments=4)

    df_exog_1 = generate_ar_df(periods=100, start_time="2020-01-01", n_segments=4, random_seed=2).rename({"target": "exog"}, axis=1)
    df_exog_2 = generate_ar_df(periods=150, start_time="2019-12-01", n_segments=4, random_seed=3).rename({"target": "regressor_1"}, axis=1)
    df_exog_3 = generate_ar_df(periods=150, start_time="2019-12-01", n_segments=4, random_seed=4).rename({"target": "regressor_2"}, axis=1)

    df_exog = pd.merge(df_exog_1, df_exog_2, on=["timestamp", "segment"], how="right")
    df_exog = pd.merge(df_exog, df_exog_3, on=["timestamp", "segment"])

    df = TSDataset.to_dataset(df)
    df_exog = TSDataset.to_dataset(df_exog)
    ts = TSDataset(df=df, freq="D", df_exog=df_exog)
    return ts


@pytest.fixture
def relevance_matrix() -> pd.DataFrame:
    table = pd.DataFrame({"regressor_1": [1, 2, 3, 4], "regressor_2": [4, 1, 5, 2], "regressor_3": [2, 4, 1, 3]})
    table.index = ["segment_1", "segment_2", "segment_3", "segment_4"]
    return table


def test_get_regressors(ts_with_complex_exog: TSDataset):
    regressors = GaleShapleyFeatureSelectionTransform._get_regressors(ts_with_complex_exog.df)
    assert sorted(regressors) == ["regressor_1", "regressor_2"]


@pytest.mark.parametrize(
    "ascending,expected",
    (
        (
            True,
            {
                "segment_1": ["regressor_1", "regressor_3", "regressor_2"],
                "segment_2": ["regressor_2", "regressor_1", "regressor_3"],
                "segment_3": ["regressor_3", "regressor_1", "regressor_2"],
                "segment_4": ["regressor_2", "regressor_3", "regressor_1"],
            }
        ),
        (
            False,
            {
                "segment_1": ["regressor_2", "regressor_3", "regressor_1"],
                "segment_2": ["regressor_3", "regressor_1", "regressor_2"],
                "segment_3": ["regressor_2", "regressor_1", "regressor_3"],
                "segment_4": ["regressor_1", "regressor_3", "regressor_2"],
            }
        ),
    )
)
def test_get_ranked_list(relevance_matrix: pd.DataFrame, ascending: bool, expected: Dict[str, List[str]]):
    result = GaleShapleyFeatureSelectionTransform._get_ranked_list(table=relevance_matrix, ascending=ascending)
    for key in expected.keys():
        assert key in result
        assert result[key] == expected[key]


@pytest.mark.parametrize(
    "ascending,expected",
    (
        (
            True,
            {
                "regressor_1": ["segment_1", "segment_2", "segment_3", "segment_4"],
                "regressor_2": ["segment_2", "segment_4", "segment_1", "segment_3"],
                "regressor_3": ["segment_3", "segment_1", "segment_4", "segment_2"],
            }
        ),
        (
            False,
            {
                "regressor_1": ["segment_4", "segment_3", "segment_2", "segment_1"],
                "regressor_2": ["segment_3", "segment_1", "segment_4", "segment_2"],
                "regressor_3": ["segment_2", "segment_4", "segment_1", "segment_3"],
            }
        ),
    )
)
def test_get_ranked_list_T(relevance_matrix: pd.DataFrame, ascending: bool, expected: Dict[str, List[str]]):
    result = GaleShapleyFeatureSelectionTransform._get_ranked_list(table=relevance_matrix.T, ascending=ascending)
    for key in expected.keys():
        assert key in result
        assert result[key] == expected[key]


@pytest.mark.parametrize(
    "top_k,n_segments,n_regressors,expected",
    (
        (20, 10, 50, 2),
        (27, 10, 40, 3),
        (15, 4, 16, 4),
        (7, 10, 50, 1),
        (30, 5, 20, 1),
    )
)
def test_compute_gale_shapley_steps_number(top_k: int, n_segments: int, n_regressors: int, expected: int):
    result = GaleShapleyFeatureSelectionTransform._compute_gale_shapley_steps_number(
        top_k=top_k, n_segments=n_segments, n_regressors=n_regressors
    )
    assert result == expected
