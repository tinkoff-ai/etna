from typing import Dict
from typing import List

import numpy as np
import pandas as pd
import pytest

from etna.analysis.feature_relevance import StatisticsRelevanceTable
from etna.datasets import TSDataset
from etna.datasets import generate_ar_df
from etna.datasets import generate_periodic_df
from etna.transforms.gale_shapley import BaseGaleShapley
from etna.transforms.gale_shapley import GaleShapleyFeatureSelectionTransform
from etna.transforms.gale_shapley import GaleShapleyMatcher
from etna.transforms.gale_shapley import RegressorGaleShapley
from etna.transforms.gale_shapley import SegmentGaleShapley


@pytest.fixture
def ts_with_large_regressors_number(random_seed) -> TSDataset:
    df = generate_periodic_df(periods=100, start_time="2020-01-01", n_segments=3, period=7, scale=10)

    exog_df = generate_periodic_df(periods=150, start_time="2020-01-01", n_segments=3, period=7).rename(
        {"target": "regressor_1"}, axis=1
    )
    for i in range(1, 4):
        tmp = generate_periodic_df(periods=150, start_time="2020-01-01", n_segments=3, period=7)
        tmp["target"] += np.random.uniform(low=-i / 5, high=i / 5, size=(450,))
        exog_df = exog_df.merge(tmp.rename({"target": f"regressor_{i + 1}"}, axis=1), on=["timestamp", "segment"])
    for i in range(4, 8):
        tmp = generate_ar_df(periods=150, start_time="2020-01-01", n_segments=3, ar_coef=[1], random_seed=i)
        exog_df = exog_df.merge(tmp.rename({"target": f"regressor_{i + 1}"}, axis=1), on=["timestamp", "segment"])

    ts = TSDataset(df=TSDataset.to_dataset(df), freq="D", df_exog=TSDataset.to_dataset(exog_df))
    return ts


@pytest.fixture
def relevance_matrix() -> pd.DataFrame:
    table = pd.DataFrame({"regressor_1": [1, 2, 3, 4], "regressor_2": [4, 1, 5, 2], "regressor_3": [2, 4, 1, 3]})
    table.index = ["segment_1", "segment_2", "segment_3", "segment_4"]
    return table


@pytest.fixture
def base_gale_shapley_player() -> BaseGaleShapley:
    base = BaseGaleShapley(name="regressor_1", ranked_candidates=["segment_1", "segment_3", "segment_2", "segment_4"])
    return base


@pytest.fixture
def regressor() -> RegressorGaleShapley:
    reg = RegressorGaleShapley(
        name="regressor_1", ranked_candidates=["segment_1", "segment_3", "segment_2", "segment_4"]
    )
    return reg


@pytest.fixture
def segment() -> SegmentGaleShapley:
    segment = SegmentGaleShapley(
        name="segment_1", ranked_candidates=["regressor_1", "regressor_2", "regressor_3", "regressor_4"]
    )
    return segment


@pytest.fixture
def matcher() -> GaleShapleyMatcher:
    segments = [
        SegmentGaleShapley(
            name="segment_1",
            ranked_candidates=["regressor_1", "regressor_2", "regressor_3"],
        ),
        SegmentGaleShapley(
            name="segment_2",
            ranked_candidates=["regressor_1", "regressor_3", "regressor_2"],
        ),
        SegmentGaleShapley(
            name="segment_3",
            ranked_candidates=["regressor_2", "regressor_3", "regressor_1"],
        ),
    ]
    regressors = [
        RegressorGaleShapley(
            name="regressor_1",
            ranked_candidates=["segment_3", "segment_1", "segment_2"],
        ),
        RegressorGaleShapley(
            name="regressor_2",
            ranked_candidates=["segment_2", "segment_3", "segment_1"],
        ),
        RegressorGaleShapley(
            name="regressor_3",
            ranked_candidates=["segment_1", "segment_2", "segment_3"],
        ),
    ]
    gsh = GaleShapleyMatcher(segments=segments, regressors=regressors)
    return gsh


@pytest.fixture
def relevance_matrix_big() -> pd.DataFrame:
    matrix = np.array([[1, 2, 3, 4, 5, 6, 7], [6, 1, 3, 4, 7, 5, 2], [1, 5, 4, 3, 2, 7, 6]])
    table = pd.DataFrame(
        matrix,
        index=["segment_1", "segment_2", "segment_3"],
        columns=[
            "regressor_1",
            "regressor_2",
            "regressor_3",
            "regressor_4",
            "regressor_5",
            "regressor_6",
            "regressor_7",
        ],
    )
    return table


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
            },
        ),
        (
            False,
            {
                "segment_1": ["regressor_2", "regressor_3", "regressor_1"],
                "segment_2": ["regressor_3", "regressor_1", "regressor_2"],
                "segment_3": ["regressor_2", "regressor_1", "regressor_3"],
                "segment_4": ["regressor_1", "regressor_3", "regressor_2"],
            },
        ),
    ),
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
            },
        ),
        (
            False,
            {
                "regressor_1": ["segment_4", "segment_3", "segment_2", "segment_1"],
                "regressor_2": ["segment_3", "segment_1", "segment_4", "segment_2"],
                "regressor_3": ["segment_2", "segment_4", "segment_1", "segment_3"],
            },
        ),
    ),
)
def test_get_ranked_list_regressors(relevance_matrix: pd.DataFrame, ascending: bool, expected: Dict[str, List[str]]):
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
    ),
)
def test_compute_gale_shapley_steps_number(top_k: int, n_segments: int, n_regressors: int, expected: int):
    result = GaleShapleyFeatureSelectionTransform._compute_gale_shapley_steps_number(
        top_k=top_k, n_segments=n_segments, n_regressors=n_regressors
    )
    assert result == expected


@pytest.mark.parametrize(
    "ranked_regressors,regressors_to_drop,expected",
    (
        (
            {
                "segment_1": ["regressor_1", "regressor_2", "regressor_3", "regressor_4"],
                "segment_2": ["regressor_3", "regressor_2", "regressor_1", "regressor_4"],
                "segment_3": ["regressor_4", "regressor_3", "regressor_1", "regressor_2"],
            },
            ["regressor_2", "regressor_3"],
            {
                "segment_1": ["regressor_1", "regressor_4"],
                "segment_2": ["regressor_1", "regressor_4"],
                "segment_3": ["regressor_4", "regressor_1"],
            },
        ),
        (
            {
                "segment_1": ["regressor_1", "regressor_2", "regressor_3", "regressor_4"],
                "segment_2": ["regressor_3", "regressor_2", "regressor_1", "regressor_4"],
                "segment_3": ["regressor_4", "regressor_3", "regressor_1", "regressor_2"],
            },
            ["regressor_2", "regressor_3", "regressor_1", "regressor_4"],
            {
                "segment_1": [],
                "segment_2": [],
                "segment_3": [],
            },
        ),
    ),
)
def test_gale_shapley_transform_update_ranking_list(
    ranked_regressors: Dict[str, List[str]], regressors_to_drop: List[str], expected: Dict[str, List[str]]
):
    result = GaleShapleyFeatureSelectionTransform._update_ranking_list(
        segment_regressors_ranking=ranked_regressors, regressors_to_drop=regressors_to_drop
    )
    for key in result:
        assert result[key] == expected[key]


def test_base_update_segment(base_gale_shapley_player: BaseGaleShapley):
    base_gale_shapley_player.update_tmp_match("segment_2")
    assert base_gale_shapley_player.tmp_match == "segment_2"
    assert base_gale_shapley_player.tmp_match_rank == 2


def test_regressor_check_segment(regressor: RegressorGaleShapley):
    assert regressor.check_segment("segment_4")
    regressor.update_tmp_match("segment_2")
    assert not regressor.check_segment("segment_4")
    assert regressor.check_segment("segment_1")


def test_segment_get_next_candidate(segment: SegmentGaleShapley):
    assert segment.get_next_candidate() == "regressor_1"
    segment.update_tmp_match("regressor_1")
    assert segment.get_next_candidate() == "regressor_2"


def test_gale_shapley_matcher_match(matcher: GaleShapleyMatcher):
    segment = matcher.segments[0]
    regressor = matcher.regressors[0]
    assert segment.tmp_match is None
    assert segment.is_available
    assert regressor.tmp_match is None
    assert regressor.is_available
    matcher.match(segment=segment, regressor=regressor)
    assert segment.tmp_match == regressor.name
    assert segment.tmp_match_rank == 0
    assert not segment.is_available
    assert regressor.tmp_match == segment.name
    assert regressor.tmp_match_rank == 1
    assert not regressor.is_available


def test_gale_shapley_matcher_break_match(matcher: GaleShapleyMatcher):
    segment = matcher.segments[0]
    regressor = matcher.regressors[0]
    assert segment.tmp_match is None
    assert segment.is_available
    assert regressor.tmp_match is None
    assert regressor.is_available
    matcher.match(segment=segment, regressor=regressor)
    matcher.break_match(segment=segment, regressor=regressor)
    assert segment.tmp_match is None
    assert segment.is_available
    assert regressor.tmp_match is None
    assert regressor.is_available


@pytest.mark.parametrize(
    "segments,regressors,expected",
    (
        (
            [
                SegmentGaleShapley(
                    name="segment_1",
                    ranked_candidates=["regressor_1", "regressor_2", "regressor_3", "regressor_4"],
                ),
                SegmentGaleShapley(
                    name="segment_2",
                    ranked_candidates=["regressor_1", "regressor_3", "regressor_2", "regressor_4"],
                ),
                SegmentGaleShapley(
                    name="segment_3",
                    ranked_candidates=["regressor_2", "regressor_4", "regressor_1", "regressor_3"],
                ),
                SegmentGaleShapley(
                    name="segment_4",
                    ranked_candidates=["regressor_3", "regressor_1", "regressor_4", "regressor_2"],
                ),
            ],
            [
                RegressorGaleShapley(
                    name="regressor_1",
                    ranked_candidates=["segment_2", "segment_1", "segment_3", "segment_4"],
                ),
                RegressorGaleShapley(
                    name="regressor_2",
                    ranked_candidates=["segment_1", "segment_2", "segment_3", "segment_4"],
                ),
                RegressorGaleShapley(
                    name="regressor_3",
                    ranked_candidates=["segment_3", "segment_2", "segment_4", "segment_1"],
                ),
                RegressorGaleShapley(
                    name="regressor_4",
                    ranked_candidates=["segment_3", "segment_1", "segment_4", "segment_2"],
                ),
            ],
            {
                "segment_1": "regressor_2",
                "segment_2": "regressor_1",
                "segment_3": "regressor_4",
                "segment_4": "regressor_3",
            },
        ),
        (
            [
                SegmentGaleShapley(
                    name="segment_1",
                    ranked_candidates=["regressor_1", "regressor_2", "regressor_3", "regressor_4"],
                ),
                SegmentGaleShapley(
                    name="segment_2",
                    ranked_candidates=["regressor_1", "regressor_2", "regressor_3", "regressor_4"],
                ),
                SegmentGaleShapley(
                    name="segment_3",
                    ranked_candidates=["regressor_1", "regressor_2", "regressor_3", "regressor_4"],
                ),
                SegmentGaleShapley(
                    name="segment_4",
                    ranked_candidates=["regressor_1", "regressor_2", "regressor_3", "regressor_4"],
                ),
            ],
            [
                RegressorGaleShapley(
                    name="regressor_1",
                    ranked_candidates=["segment_2", "segment_1", "segment_3", "segment_4"],
                ),
                RegressorGaleShapley(
                    name="regressor_2",
                    ranked_candidates=["segment_1", "segment_2", "segment_3", "segment_4"],
                ),
                RegressorGaleShapley(
                    name="regressor_3",
                    ranked_candidates=["segment_3", "segment_2", "segment_4", "segment_1"],
                ),
                RegressorGaleShapley(
                    name="regressor_4",
                    ranked_candidates=["segment_3", "segment_1", "segment_4", "segment_2"],
                ),
            ],
            {
                "segment_1": "regressor_2",
                "segment_2": "regressor_1",
                "segment_3": "regressor_3",
                "segment_4": "regressor_4",
            },
        ),
        (
            [
                SegmentGaleShapley(
                    name="segment_1",
                    ranked_candidates=["regressor_1", "regressor_5", "regressor_2", "regressor_4", "regressor_3"],
                ),
                SegmentGaleShapley(
                    name="segment_2",
                    ranked_candidates=["regressor_5", "regressor_2", "regressor_3", "regressor_4", "regressor_1"],
                ),
                SegmentGaleShapley(
                    name="segment_3",
                    ranked_candidates=["regressor_1", "regressor_2", "regressor_3", "regressor_4", "regressor_5"],
                ),
            ],
            [
                RegressorGaleShapley(
                    name="regressor_1",
                    ranked_candidates=["segment_3", "segment_1", "segment_2"],
                ),
                RegressorGaleShapley(
                    name="regressor_2",
                    ranked_candidates=["segment_3", "segment_2", "segment_1"],
                ),
                RegressorGaleShapley(
                    name="regressor_3",
                    ranked_candidates=["segment_3", "segment_1", "segment_2"],
                ),
                RegressorGaleShapley(
                    name="regressor_4",
                    ranked_candidates=["segment_1", "segment_2", "segment_3"],
                ),
                RegressorGaleShapley(
                    name="regressor_5",
                    ranked_candidates=["segment_1", "segment_3", "segment_2"],
                ),
            ],
            {
                "segment_1": "regressor_5",
                "segment_2": "regressor_2",
                "segment_3": "regressor_1",
            },
        ),
    ),
)
def test_gale_shapley_result(
    segments: List[SegmentGaleShapley],
    regressors: List[RegressorGaleShapley],
    expected: Dict[str, str],
):
    matcher = GaleShapleyMatcher(segments=segments, regressors=regressors)
    matches = matcher()
    for k, v in expected.items():
        assert k in matches
        assert matches[k] == v


@pytest.mark.parametrize(
    "segment_regressor_ranking,regressor_segments_ranking,expected",
    (
        (
            {
                "segment_1": ["regressor_1", "regressor_2", "regressor_3", "regressor_4"],
                "segment_2": ["regressor_1", "regressor_3", "regressor_2", "regressor_4"],
                "segment_3": ["regressor_2", "regressor_4", "regressor_1", "regressor_3"],
                "segment_4": ["regressor_3", "regressor_1", "regressor_4", "regressor_2"],
            },
            {
                "regressor_1": ["segment_2", "segment_1", "segment_3", "segment_4"],
                "regressor_2": ["segment_1", "segment_2", "segment_3", "segment_4"],
                "regressor_3": ["segment_3", "segment_2", "segment_4", "segment_1"],
                "regressor_4": ["segment_3", "segment_1", "segment_4", "segment_2"],
            },
            {
                "segment_1": "regressor_2",
                "segment_2": "regressor_1",
                "segment_3": "regressor_4",
                "segment_4": "regressor_3",
            },
        ),
        (
            {
                "segment_1": ["regressor_1", "regressor_2", "regressor_3", "regressor_4"],
                "segment_2": ["regressor_1", "regressor_2", "regressor_3", "regressor_4"],
                "segment_3": ["regressor_1", "regressor_2", "regressor_3", "regressor_4"],
                "segment_4": ["regressor_1", "regressor_2", "regressor_3", "regressor_4"],
            },
            {
                "regressor_1": ["segment_2", "segment_1", "segment_3", "segment_4"],
                "regressor_2": ["segment_1", "segment_2", "segment_3", "segment_4"],
                "regressor_3": ["segment_3", "segment_2", "segment_4", "segment_1"],
                "regressor_4": ["segment_3", "segment_1", "segment_4", "segment_2"],
            },
            {
                "segment_1": "regressor_2",
                "segment_2": "regressor_1",
                "segment_3": "regressor_3",
                "segment_4": "regressor_4",
            },
        ),
        (
            {
                "segment_1": ["regressor_1", "regressor_5", "regressor_2", "regressor_4", "regressor_3"],
                "segment_2": ["regressor_5", "regressor_2", "regressor_3", "regressor_4", "regressor_1"],
                "segment_3": ["regressor_1", "regressor_2", "regressor_3", "regressor_4", "regressor_5"],
            },
            {
                "regressor_1": ["segment_3", "segment_1", "segment_2"],
                "regressor_2": ["segment_3", "segment_2", "segment_1"],
                "regressor_3": ["segment_3", "segment_1", "segment_2"],
                "regressor_4": ["segment_1", "segment_2", "segment_3"],
                "regressor_5": ["segment_1", "segment_3", "segment_2"],
            },
            {
                "segment_1": "regressor_5",
                "segment_2": "regressor_2",
                "segment_3": "regressor_1",
            },
        ),
    ),
)
def test_gale_shapley_transform_gale_shapley_iteration(
    segment_regressor_ranking: Dict[str, List[str]],
    regressor_segments_ranking: Dict[str, List[str]],
    expected: Dict[str, str],
):
    GaleShapleyFeatureSelectionTransform._gale_shapley_iteration(
        segment_regressors_ranking=segment_regressor_ranking, regressor_segments_ranking=regressor_segments_ranking
    )


@pytest.mark.parametrize(
    "matches,n,greater_is_better,expected",
    (
        (
            {
                "segment_1": "regressor_4",
                "segment_2": "regressor_7",
                "segment_3": "regressor_5",
            },
            2,
            False,
            ["regressor_5", "regressor_7"],
        ),
        (
            {
                "segment_1": "regressor_4",
                "segment_2": "regressor_7",
                "segment_3": "regressor_5",
            },
            1,
            True,
            ["regressor_4"],
        ),
        (
            {
                "segment_1": "regressor_3",
                "segment_2": "regressor_2",
                "segment_3": "regressor_1",
            },
            2,
            False,
            ["regressor_1", "regressor_2"],
        ),
        (
            {
                "segment_1": "regressor_3",
                "segment_2": "regressor_2",
                "segment_3": "regressor_1",
            },
            3,
            False,
            ["regressor_1", "regressor_2", "regressor_3"],
        ),
    ),
)
def test_gale_shapley_transform_process_last_step(
    matches: Dict[str, str], n: int, greater_is_better: bool, expected: List[str], relevance_matrix_big: pd.DataFrame
):
    result = GaleShapleyFeatureSelectionTransform._process_last_step(
        matches=matches, relevance_table=relevance_matrix_big, n=n, greater_is_better=greater_is_better
    )
    assert sorted(result) == sorted(expected)


@pytest.mark.parametrize("use_rank", (True, False))
@pytest.mark.parametrize("top_k", (2, 3, 5, 6, 7))
def test_gale_shapley_transform_fit(ts_with_large_regressors_number: TSDataset, top_k: int, use_rank: bool):
    df = ts_with_large_regressors_number.df
    transform = GaleShapleyFeatureSelectionTransform(
        relevance_table=StatisticsRelevanceTable(), top_k=top_k, use_rank=use_rank
    )
    transform.fit(df=df)


def test_gale_shapley_transform_fit_transform(ts_with_large_regressors_number: TSDataset):
    df = ts_with_large_regressors_number.df
    transform = GaleShapleyFeatureSelectionTransform(
        relevance_table=StatisticsRelevanceTable(), top_k=5, use_rank=False
    )
    transformed = transform.fit_transform(df=df)
    assert set(transformed.columns.get_level_values("feature")) == {
        "target",
        "regressor_1",
        "regressor_2",
        "regressor_3",
        "regressor_4",
        "regressor_5",
    }
