from copy import deepcopy

import pandas as pd
import pytest


def expected_ts(ts, df_idx, exog_idx):
    expected_slice = deepcopy(ts)
    expected_slice.df = expected_slice.df.loc[df_idx]
    expected_slice.df_exog = expected_slice.df_exog.loc[exog_idx]
    return expected_slice


@pytest.mark.parametrize("idx", [(1), ("2020-01-02")])
def test_one_point_slice(example_reg_tsds, idx):
    expected_slice = expected_ts(example_reg_tsds, df_idx=["2020-01-02"], exog_idx=slice(None))
    ts_slice = example_reg_tsds[idx]
    assert ts_slice == expected_slice


@pytest.mark.parametrize("idx", [(slice(1, 2)), (slice("2020-01-02", "2020-01-03"))])
def test_two_points_slice(example_reg_tsds, idx):
    expected_slice = expected_ts(example_reg_tsds, df_idx=slice("2020-01-02", "2020-01-03"), exog_idx=slice(None))
    ts_slice = example_reg_tsds[idx]
    assert ts_slice == expected_slice


@pytest.mark.parametrize("idx", [(slice(1, 2)), (slice("2020-01-02", "2020-01-03"))])
def test_two_points_feature_slice(example_reg_tsds, idx, feature="target"):
    expected_slice = expected_ts(
        example_reg_tsds,
        df_idx=(pd.IndexSlice["2020-01-02":"2020-01-03"], pd.IndexSlice[:, feature]),
        exog_idx=slice(None),
    )
    ts_slice = example_reg_tsds[idx, feature]
    assert ts_slice == expected_slice


@pytest.mark.parametrize("idx", [(slice(1, 2)), (slice("2020-01-02", "2020-01-03"))])
def test_two_points_segment_feature_slice(example_reg_tsds, idx, segment="segment_1", feature="target"):
    expected_slice = expected_ts(
        example_reg_tsds,
        df_idx=(pd.IndexSlice["2020-01-02":"2020-01-03"], pd.IndexSlice[[segment], [feature]]),
        exog_idx=(pd.IndexSlice[:], pd.IndexSlice[segment, :]),
    )
    ts_slice = example_reg_tsds[idx, segment, feature]
    assert ts_slice == expected_slice


def test_head_default(example_reg_tsds):
    expected_slice = expected_ts(
        example_reg_tsds,
        df_idx=slice("2020-01-01", "2020-01-05"),
        exog_idx=slice(None),
    )
    ts_slice = example_reg_tsds.head()
    assert ts_slice == expected_slice


def test_tail_default(example_reg_tsds):
    expected_slice = expected_ts(
        example_reg_tsds,
        df_idx=slice("2020-04-05", "2020-04-09"),
        exog_idx=slice(None),
    )
    ts_slice = example_reg_tsds.tail()
    assert ts_slice == expected_slice
