from copy import deepcopy
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import pandas as pd
import pytest

from etna.transforms.datetime_flags import TimeFlagsTransform

INIT_PARAMS_TEMPLATE = {
    "minute_in_hour_number": False,
    "fifteen_minutes_in_hour_number": False,
    "hour_number": False,
    "half_hour_number": False,
    "half_day_number": False,
    "one_third_day_number": False,
}


@pytest.fixture
def dateflags_true_df() -> pd.DataFrame:
    """
    Generate dataset for TimeFlags feature

    Returns
    -------
    dataset with timestamp column and columns true_minute_in_hour_number, true_fifteen_minutes_in_hour_number,
    true_half_hour_number, true_hour_number, true_half_day_number, true_one_third_day_number that contain
    true answers for corresponding features
    """
    dataframes = [
        pd.DataFrame({"timestamp": pd.date_range("2020-06-01", "2021-06-01", freq="5 min")}) for i in range(5)
    ]

    for i in range(len(dataframes)):
        df = dataframes[i]
        df["minute_in_hour_number"] = df["timestamp"].dt.minute
        df["fifteen_minutes_in_hour_number"] = df["minute_in_hour_number"] // 15
        df["half_hour_number"] = df["minute_in_hour_number"] // 30

        df["hour_number"] = df["timestamp"].dt.hour
        df["half_day_number"] = df["hour_number"] // 12
        df["one_third_day_number"] = df["hour_number"] // 8

        df.columns = df.columns.map(lambda col: "regressor_" + col if col != "timestamp" else col)
        df["segment"] = f"segment_{i}"
        df["target"] = 2

    result = pd.concat(dataframes, ignore_index=True)
    result = result.pivot(index="timestamp", columns="segment")
    result = result.reorder_levels([1, 0], axis=1)
    result = result.sort_index(axis=1)
    result.columns.names = ["segment", "feature"]

    return result


@pytest.fixture
def train_df() -> pd.DataFrame:
    """
    Generate dataset without dateflags
    """
    dataframes = [
        pd.DataFrame({"timestamp": pd.date_range("2020-06-01", "2021-06-01", freq="5 min")}) for i in range(5)
    ]

    for i in range(len(dataframes)):
        df = dataframes[i]
        df["segment"] = f"segment_{i}"
        df["target"] = 2

    result = pd.concat(dataframes, ignore_index=True)
    result = result.pivot(index="timestamp", columns="segment")
    result = result.reorder_levels([1, 0], axis=1)
    result = result.sort_index(axis=1)
    result.columns.names = ["segment", "feature"]

    return result


def test_interface_incorrect_args():
    """This test checks feature's behavior in case of incorrect argument set"""
    with pytest.raises(ValueError):
        _ = TimeFlagsTransform(
            minute_in_hour_number=False,
            fifteen_minutes_in_hour_number=False,
            half_hour_number=False,
            hour_number=False,
            half_day_number=False,
            one_third_day_number=False,
        )


@pytest.mark.parametrize(
    "true_params",
    (
        ["minute_in_hour_number"],
        ["fifteen_minutes_in_hour_number"],
        ["hour_number"],
        ["half_hour_number"],
        ["half_day_number"],
        ["one_third_day_number"],
        [
            "minute_in_hour_number",
            "fifteen_minutes_in_hour_number",
            "hour_number",
            "half_hour_number",
            "half_day_number",
            "one_third_day_number",
        ],
    ),
)
def test_interface_correct_args(true_params: List[str], train_df: pd.DataFrame):
    """This test checks that feature generates all the expected columns and no unexpected ones in transform"""
    init_params = deepcopy(INIT_PARAMS_TEMPLATE)
    test_segs = train_df.columns.get_level_values(0).unique()
    for key in true_params:
        init_params[key] = True
    transform = TimeFlagsTransform(**init_params)
    result = transform.fit_transform(df=train_df.copy())

    assert sorted(test_segs) == sorted(result.columns.get_level_values(0).unique())
    assert sorted(result.columns.names) == ["feature", "segment"]

    true_params = ["regressor_" + param for param in true_params]
    for seg in result.columns.get_level_values(0).unique():
        tmp_df = result[seg]
        assert sorted(list(tmp_df.columns)) == sorted(true_params + ["target"])
        for param in true_params:
            assert tmp_df[param].dtype == "category"


@pytest.mark.parametrize(
    "true_params",
    (
        {"minute_in_hour_number": True},
        {"fifteen_minutes_in_hour_number": True},
        {"hour_number": True},
        {"half_hour_number": True},
        {"half_day_number": True},
        {"one_third_day_number": True},
    ),
)
def test_feature_values(
    true_params: Dict[str, Union[bool, Tuple[int, int]]], train_df: pd.DataFrame, dateflags_true_df: pd.DataFrame
):
    """This test checks that feature generates correct values"""
    init_params = deepcopy(INIT_PARAMS_TEMPLATE)
    init_params.update(true_params)
    transform = TimeFlagsTransform(**init_params)
    result = transform.fit_transform(df=train_df.copy())

    segments_true = dateflags_true_df.columns.get_level_values(0).unique()
    segment_result = result.columns.get_level_values(0).unique()

    assert sorted(segment_result) == sorted(segments_true)

    true_params = ["regressor_" + param for param in true_params.keys()]
    for seg in segment_result:
        segment_true = dateflags_true_df[seg]
        true_df = segment_true[true_params + ["target"]].sort_index(axis=1)
        result_df = result[seg].sort_index(axis=1)
        assert (true_df == result_df).all().all()
