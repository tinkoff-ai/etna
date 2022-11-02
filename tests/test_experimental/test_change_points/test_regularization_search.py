import pytest
from ruptures import Binseg

from etna.datasets import TSDataset
from etna.datasets import generate_ar_df
from etna.experimental.change_points import get_ruptures_regularization
from etna.experimental.change_points.regularization_search import _get_n_bkps


@pytest.fixture
def simple_change_points_ts():
    df = generate_ar_df(periods=125, start_time="2021-05-20", n_segments=3, freq="D", random_seed=42)
    df_ts_format = TSDataset.to_dataset(df)
    return TSDataset(df_ts_format, freq="D")


@pytest.mark.parametrize(
    "segment,params,expected",
    (
        ("segment_0", {"pen": 20}, 6),
        ("segment_0", {"epsilon": 20}, 24),
        ("segment_1", {"pen": 10}, 7),
        ("segment_1", {"epsilon": 100}, 12),
        ("segment_2", {"pen": 2}, 14),
        ("segment_2", {"epsilon": 200}, 6),
    ),
)
def test_get_n_bkps(segment, params, expected, simple_change_points_ts):
    series = simple_change_points_ts[:, segment, "target"]
    assert _get_n_bkps(series, Binseg(), **params) == expected


@pytest.mark.parametrize(
    "n_bkps,mode",
    (
        ({"segment_0": 3, "segment_1": 14, "segment_2": 19}, "pen"),
        ({"segment_0": 5, "segment_1": 2, "segment_2": 8}, "epsilon"),
        ({"segment_0": 11, "segment_1": 18, "segment_2": 4}, "pen"),
        ({"segment_0": 18, "segment_1": 21, "segment_2": 7}, "epsilon"),
    ),
)
def test_get_regularization(n_bkps, mode, simple_change_points_ts):
    in_column = "target"
    res = get_ruptures_regularization(
        simple_change_points_ts, in_column=in_column, change_point_model=Binseg(), n_bkps=n_bkps, mode=mode
    )
    assert sorted(res.keys()) == sorted(simple_change_points_ts.to_pandas().columns.get_level_values(0).unique())
    for seg in res.keys():
        series = simple_change_points_ts[:, seg, in_column]
        answer = _get_n_bkps(series, Binseg(), **{mode: res[seg][mode]})
        assert answer == n_bkps[seg]


@pytest.mark.parametrize(
    "n_bkps,mode",
    (
        ({"segment_0": 3, "segment_1": 34, "segment_2": 19}, "pen"),
        ({"segment_0": 45, "segment_1": 2, "segment_2": 8}, "epsilon"),
    ),
)
def test_fail_get_regularization_high(n_bkps, mode, simple_change_points_ts):
    in_column = "target"
    with pytest.raises(ValueError, match="Impossible number of changepoints. Please, decrease n_bkps value."):
        _ = get_ruptures_regularization(
            simple_change_points_ts, in_column=in_column, change_point_model=Binseg(), n_bkps=n_bkps, mode=mode
        )


@pytest.mark.parametrize(
    "n_bkps,mode",
    (
        ({"segment_0": 3, "segment_1": 1, "segment_2": 19}, "pen"),
        ({"segment_0": 1, "segment_1": 2, "segment_2": 8}, "epsilon"),
    ),
)
def test_fail_get_regularization_low(n_bkps, mode, simple_change_points_ts):
    in_column = "target"
    with pytest.raises(
        ValueError, match="Impossible number of changepoints. Please, increase max_value or increase n_bkps value."
    ):
        _ = get_ruptures_regularization(
            simple_change_points_ts,
            in_column=in_column,
            change_point_model=Binseg(),
            n_bkps=n_bkps,
            mode=mode,
            max_value=1,
        )
