from typing import Dict
from typing import List

import numpy as np
import pandas as pd
import pytest

from etna.datasets import TSDataset
from etna.datasets import generate_ar_df
from etna.transforms.gale_shapley import BaseGaleShapley
from etna.transforms.gale_shapley import GaleShapleyFeatureSelectionTransform
from etna.transforms.gale_shapley import GaleShapleyMatcher
from etna.transforms.gale_shapley import RegressorGaleShapley
from etna.transforms.gale_shapley import SegmentGaleShapley


@pytest.fixture
def ts_with_complex_exog() -> TSDataset:
    df = generate_ar_df(periods=100, start_time="2020-01-01", n_segments=4)

    df_exog_1 = generate_ar_df(periods=100, start_time="2020-01-01", n_segments=4, random_seed=2).rename(
        {"target": "exog"}, axis=1
    )
    df_exog_2 = generate_ar_df(periods=150, start_time="2019-12-01", n_segments=4, random_seed=3).rename(
        {"target": "regressor_1"}, axis=1
    )
    df_exog_3 = generate_ar_df(periods=150, start_time="2019-12-01", n_segments=4, random_seed=4).rename(
        {"target": "regressor_2"}, axis=1
    )

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


def test_gale_shapley_matcher_gale_shapley_iteration(matcher: GaleShapleyMatcher):
    available_segments = matcher._get_available_segments()
    matcher._gale_shapley_iteration(available_segments=available_segments)
    assert not matcher.segments[0].is_available
    assert matcher.segments[0].tmp_match == "regressor_1"
    assert matcher.segments[1].is_available
    assert matcher.segments[1].tmp_match is None
    assert not matcher.segments[2].is_available
    assert matcher.segments[2].tmp_match == "regressor_2"
    available_segments = matcher._get_available_segments()
    matcher._gale_shapley_iteration(available_segments=available_segments)
    assert not matcher.segments[0].is_available
    assert matcher.segments[0].tmp_match == "regressor_1"
    assert not matcher.segments[1].is_available
    assert matcher.segments[1].tmp_match == "regressor_3"
    assert not matcher.segments[2].is_available
    assert matcher.segments[2].tmp_match == "regressor_2"


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


@pytest.fixture
def relevance_table() -> pd.DataFrame:
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
    "matches,n,expected",
    (
        (
            {
                "segment_1": "regressor_4",
                "segment_2": "regressor_7",
                "segment_3": "regressor_5",
            },
            2,
            ["regressor_5", "regressor_7"],
        ),
        (
            {
                "segment_1": "regressor_3",
                "segment_2": "regressor_2",
                "segment_3": "regressor_1",
            },
            2,
            ["regressor_1", "regressor_2"],
        ),
        (
            {
                "segment_1": "regressor_3",
                "segment_2": "regressor_2",
                "segment_3": "regressor_1",
            },
            3,
            ["regressor_1", "regressor_2", "regressor_3"],
        ),
    ),
)
def test_gale_shapley_transform_process_last_step(
    matches: Dict[str, str], n: int, expected: List[str], relevance_table: pd.DataFrame
):
    result = GaleShapleyFeatureSelectionTransform._process_last_step(
        matches=matches, relevance_table=relevance_table, n=n
    )
    assert sorted(result) == sorted(expected)
