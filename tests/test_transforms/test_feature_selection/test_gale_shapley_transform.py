from typing import Dict
from typing import List

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from etna.analysis.feature_relevance import ModelRelevanceTable
from etna.analysis.feature_relevance import StatisticsRelevanceTable
from etna.datasets import TSDataset
from etna.datasets import generate_ar_df
from etna.datasets import generate_periodic_df
from etna.transforms.feature_selection import GaleShapleyFeatureSelectionTransform
from etna.transforms.feature_selection.gale_shapley import BaseGaleShapley
from etna.transforms.feature_selection.gale_shapley import FeatureGaleShapley
from etna.transforms.feature_selection.gale_shapley import GaleShapleyMatcher
from etna.transforms.feature_selection.gale_shapley import SegmentGaleShapley
from tests.test_transforms.utils import assert_sampling_is_valid
from tests.test_transforms.utils import assert_transformation_equals_loaded_original


@pytest.fixture
def ts_with_exog_galeshapley(random_seed) -> TSDataset:
    np.random.seed(random_seed)

    periods = 30
    df_1 = pd.DataFrame({"timestamp": pd.date_range("2020-01-15", periods=periods)})
    df_1["segment"] = "segment_1"
    df_1["target"] = np.random.uniform(10, 20, size=periods)

    df_2 = pd.DataFrame({"timestamp": pd.date_range("2020-01-15", periods=periods)})
    df_2["segment"] = "segment_2"
    df_2["target"] = np.random.uniform(-15, 5, size=periods)

    df = pd.concat([df_1, df_2]).reset_index(drop=True)
    df = TSDataset.to_dataset(df)
    tsds = TSDataset(df, freq="D")
    df = tsds.to_pandas(flatten=True)
    df_exog = df.copy().drop(columns=["target"])
    df_exog["weekday"] = df_exog["timestamp"].dt.weekday
    df_exog["monthday"] = df_exog["timestamp"].dt.day
    df_exog["month"] = df_exog["timestamp"].dt.month
    df_exog["year"] = df_exog["timestamp"].dt.year
    ts = TSDataset(df=TSDataset.to_dataset(df), df_exog=TSDataset.to_dataset(df_exog), freq="D")
    return ts


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

    ts = TSDataset(df=TSDataset.to_dataset(df), freq="D", df_exog=TSDataset.to_dataset(exog_df), known_future="all")
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
def feature() -> FeatureGaleShapley:
    reg = FeatureGaleShapley(name="regressor_1", ranked_candidates=["segment_1", "segment_3", "segment_2", "segment_4"])
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
    features = [
        FeatureGaleShapley(
            name="regressor_1",
            ranked_candidates=["segment_3", "segment_1", "segment_2"],
        ),
        FeatureGaleShapley(
            name="regressor_2",
            ranked_candidates=["segment_2", "segment_3", "segment_1"],
        ),
        FeatureGaleShapley(
            name="regressor_3",
            ranked_candidates=["segment_1", "segment_2", "segment_3"],
        ),
    ]
    gsh = GaleShapleyMatcher(segments=segments, features=features)
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
def test_get_ranked_list_features(relevance_matrix: pd.DataFrame, ascending: bool, expected: Dict[str, List[str]]):
    result = GaleShapleyFeatureSelectionTransform._get_ranked_list(table=relevance_matrix.T, ascending=ascending)
    for key in expected.keys():
        assert key in result
        assert result[key] == expected[key]


@pytest.mark.parametrize(
    "top_k,n_segments,n_features,expected",
    (
        (20, 10, 50, 2),
        (27, 10, 40, 3),
        (15, 4, 16, 4),
        (7, 10, 50, 1),
        (30, 5, 20, 1),
    ),
)
def test_compute_gale_shapley_steps_number(top_k: int, n_segments: int, n_features: int, expected: int):
    result = GaleShapleyFeatureSelectionTransform._compute_gale_shapley_steps_number(
        top_k=top_k, n_segments=n_segments, n_features=n_features
    )
    assert result == expected


@pytest.mark.parametrize(
    "ranked_features,features_to_drop,expected",
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
    ranked_features: Dict[str, List[str]], features_to_drop: List[str], expected: Dict[str, List[str]]
):
    result = GaleShapleyFeatureSelectionTransform._update_ranking_list(
        segment_features_ranking=ranked_features, features_to_drop=features_to_drop
    )
    for key in result:
        assert result[key] == expected[key]


def test_base_update_segment(base_gale_shapley_player: BaseGaleShapley):
    base_gale_shapley_player.update_tmp_match("segment_2")
    assert base_gale_shapley_player.tmp_match == "segment_2"
    assert base_gale_shapley_player.tmp_match_rank == 2


def test_feature_check_segment(feature: FeatureGaleShapley):
    assert feature.check_segment("segment_4")
    feature.update_tmp_match("segment_2")
    assert not feature.check_segment("segment_4")
    assert feature.check_segment("segment_1")


def test_segment_get_next_candidate(segment: SegmentGaleShapley):
    assert segment.get_next_candidate() == "regressor_1"
    segment.update_tmp_match("regressor_1")
    assert segment.get_next_candidate() == "regressor_2"


def test_gale_shapley_matcher_match(matcher: GaleShapleyMatcher):
    segment = matcher.segments[0]
    feature = matcher.features[0]
    assert segment.tmp_match is None
    assert segment.is_available
    assert feature.tmp_match is None
    assert feature.is_available
    matcher.match(segment=segment, feature=feature)
    assert segment.tmp_match == feature.name
    assert segment.tmp_match_rank == 0
    assert not segment.is_available
    assert feature.tmp_match == segment.name
    assert feature.tmp_match_rank == 1
    assert not feature.is_available


def test_gale_shapley_matcher_break_match(matcher: GaleShapleyMatcher):
    segment = matcher.segments[0]
    feature = matcher.features[0]
    assert segment.tmp_match is None
    assert segment.is_available
    assert feature.tmp_match is None
    assert feature.is_available
    matcher.match(segment=segment, feature=feature)
    matcher.break_match(segment=segment, feature=feature)
    assert segment.tmp_match is None
    assert segment.is_available
    assert feature.tmp_match is None
    assert feature.is_available


@pytest.mark.parametrize(
    "segments,features,expected",
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
                FeatureGaleShapley(
                    name="regressor_1",
                    ranked_candidates=["segment_2", "segment_1", "segment_3", "segment_4"],
                ),
                FeatureGaleShapley(
                    name="regressor_2",
                    ranked_candidates=["segment_1", "segment_2", "segment_3", "segment_4"],
                ),
                FeatureGaleShapley(
                    name="regressor_3",
                    ranked_candidates=["segment_3", "segment_2", "segment_4", "segment_1"],
                ),
                FeatureGaleShapley(
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
                FeatureGaleShapley(
                    name="regressor_1",
                    ranked_candidates=["segment_2", "segment_1", "segment_3", "segment_4"],
                ),
                FeatureGaleShapley(
                    name="regressor_2",
                    ranked_candidates=["segment_1", "segment_2", "segment_3", "segment_4"],
                ),
                FeatureGaleShapley(
                    name="regressor_3",
                    ranked_candidates=["segment_3", "segment_2", "segment_4", "segment_1"],
                ),
                FeatureGaleShapley(
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
                FeatureGaleShapley(
                    name="regressor_1",
                    ranked_candidates=["segment_3", "segment_1", "segment_2"],
                ),
                FeatureGaleShapley(
                    name="regressor_2",
                    ranked_candidates=["segment_3", "segment_2", "segment_1"],
                ),
                FeatureGaleShapley(
                    name="regressor_3",
                    ranked_candidates=["segment_3", "segment_1", "segment_2"],
                ),
                FeatureGaleShapley(
                    name="regressor_4",
                    ranked_candidates=["segment_1", "segment_2", "segment_3"],
                ),
                FeatureGaleShapley(
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
    features: List[FeatureGaleShapley],
    expected: Dict[str, str],
):
    matcher = GaleShapleyMatcher(segments=segments, features=features)
    matches = matcher()
    for k, v in expected.items():
        assert k in matches
        assert matches[k] == v


@pytest.mark.parametrize(
    "segment_feature_ranking,feature_segments_ranking,expected",
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
    segment_feature_ranking: Dict[str, List[str]],
    feature_segments_ranking: Dict[str, List[str]],
    expected: Dict[str, str],
):
    GaleShapleyFeatureSelectionTransform._gale_shapley_iteration(
        segment_features_ranking=segment_feature_ranking, feature_segments_ranking=feature_segments_ranking
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
    ts = ts_with_large_regressors_number
    transform = GaleShapleyFeatureSelectionTransform(
        relevance_table=StatisticsRelevanceTable(), top_k=top_k, use_rank=use_rank
    )
    transform.fit(ts)


def test_gale_shapley_transform_fit_transform(ts_with_large_regressors_number: TSDataset):
    ts = ts_with_large_regressors_number
    transform = GaleShapleyFeatureSelectionTransform(
        relevance_table=StatisticsRelevanceTable(), top_k=5, use_rank=False
    )
    transformed = transform.fit_transform(ts)
    assert set(transformed.columns.get_level_values("feature")) == {
        "target",
        "regressor_1",
        "regressor_2",
        "regressor_3",
        "regressor_4",
        "regressor_5",
    }


@pytest.mark.parametrize("use_rank", (True, False))
@pytest.mark.parametrize("top_k", (2, 3, 5, 6, 7))
def test_gale_shapley_transform_fit_model_based(ts_with_large_regressors_number: TSDataset, top_k: int, use_rank: bool):
    ts = ts_with_large_regressors_number
    transform = GaleShapleyFeatureSelectionTransform(
        relevance_table=ModelRelevanceTable(), top_k=top_k, use_rank=use_rank, model=RandomForestRegressor()
    )
    transform.fit(ts)


@pytest.mark.xfail
def test_fit_transform_with_nans(regressor_exog_weekend):
    transform = GaleShapleyFeatureSelectionTransform(
        relevance_table=StatisticsRelevanceTable(), top_k=5, use_rank=False
    )
    transform.fit_transform(regressor_exog_weekend)


def test_work_with_non_regressors(ts_with_exog):
    selector = GaleShapleyFeatureSelectionTransform(
        relevance_table=StatisticsRelevanceTable(), top_k=3, use_rank=False, features_to_use="all"
    )
    selector.fit_transform(ts_with_exog)


@pytest.mark.parametrize(
    "transform",
    [
        GaleShapleyFeatureSelectionTransform(
            relevance_table=ModelRelevanceTable(), top_k=3, use_rank=False, model=RandomForestRegressor(random_state=42)
        ),
        GaleShapleyFeatureSelectionTransform(relevance_table=StatisticsRelevanceTable(), top_k=3, use_rank=False),
    ],
)
def test_save_load(transform, ts_with_large_regressors_number):
    assert_transformation_equals_loaded_original(transform=transform, ts=ts_with_large_regressors_number)


def test_right_number_features_with_integer_division(ts_with_exog_galeshapley):
    top_k = len(ts_with_exog_galeshapley.segments)
    transform = GaleShapleyFeatureSelectionTransform(relevance_table=StatisticsRelevanceTable(), top_k=top_k)

    transform.fit(ts_with_exog_galeshapley)
    ts = transform.transform(ts_with_exog_galeshapley)

    remaining_columns = ts.columns.get_level_values("feature").unique().tolist()
    assert len(remaining_columns) == top_k + 1


@pytest.mark.parametrize(
    "transform",
    [
        GaleShapleyFeatureSelectionTransform(
            relevance_table=ModelRelevanceTable(), top_k=3, use_rank=False, model=RandomForestRegressor(random_state=42)
        ),
        GaleShapleyFeatureSelectionTransform(relevance_table=StatisticsRelevanceTable(), top_k=3, use_rank=False),
    ],
)
def test_params_to_tune(transform, ts_with_large_regressors_number):
    ts = ts_with_large_regressors_number
    assert len(transform.params_to_tune()) > 0
    assert_sampling_is_valid(transform=transform, ts=ts)
