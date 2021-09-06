from copy import deepcopy
from datetime import timedelta
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import pandas as pd
import pytest

from etna.transforms.datetime_flags import DateFlagsTransform

SPECIAL_DAYS = [1, 4]
INIT_PARAMS_TEMPLATE = {
    "day_number_in_week": False,
    "day_number_in_month": False,
    "week_number_in_year": False,
    "week_number_in_month": False,
    "month_number_in_year": False,
    "special_days_in_week": (),
    "special_days_in_month": (),
    "year_number": False,
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
    dataframes = [pd.DataFrame({"timestamp": pd.date_range("2010-06-01", "2021-06-01", freq="3h")}) for i in range(5)]

    for i in range(len(dataframes)):
        df = dataframes[i]
        df["day_number_in_week"] = df["timestamp"].dt.weekday
        df["day_number_in_month"] = df["timestamp"].dt.day
        df["week_number_in_year"] = df["timestamp"].dt.week
        df["month_number_in_year"] = df["timestamp"].dt.month
        df["year_number"] = df["timestamp"].dt.year
        df["week_number_in_month"] = df["timestamp"].apply(
            lambda x: int(x.weekday() < (x - timedelta(days=x.day - 1)).weekday()) + (x.day - 1) // 7 + 1
        )
        df["special_days_in_week"] = df["day_number_in_week"].apply(lambda x: x in SPECIAL_DAYS)
        df["special_days_in_month"] = df["day_number_in_month"].apply(lambda x: x in SPECIAL_DAYS)

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
    dataframes = [pd.DataFrame({"timestamp": pd.date_range("2010-06-01", "2021-06-01", freq="3h")}) for i in range(5)]

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


def test_invalid_arguments_configuration():
    """This test check DateFlagsFeature's behavior in case of invalid set of params"""
    with pytest.raises(ValueError):
        _ = DateFlagsTransform(
            day_number_in_month=False,
            day_number_in_week=False,
            week_number_in_month=False,
            week_number_in_year=False,
            month_number_in_year=False,
            year_number=False,
            special_days_in_week=(),
            special_days_in_month=(),
        )


def test_repr_default():
    """This test checks that __repr__ method works fine."""
    transform_class_repr = "DateFlagsTransform"
    transform = DateFlagsTransform(
        day_number_in_week=True,
        day_number_in_month=True,
        week_number_in_month=False,
        week_number_in_year=False,
        month_number_in_year=True,
        year_number=True,
        special_days_in_week=(1, 2),
        special_days_in_month=(12,),
    )
    transform_repr = transform.__repr__()
    true_repr = (
        f"{transform_class_repr}(day_number_in_week = True, day_number_in_month = True, week_number_in_month = False, "
        f"week_number_in_year = False, month_number_in_year = True, year_number = True, special_days_in_week = (1, 2), "
        f"special_days_in_month = (12,), )"
    )
    assert transform_repr == true_repr


@pytest.mark.parametrize(
    "true_params",
    (
        ["day_number_in_week"],
        ["day_number_in_month"],
        ["week_number_in_year"],
        ["week_number_in_month"],
        ["month_number_in_year"],
        ["year_number"],
        [
            "day_number_in_week",
            "day_number_in_month",
            "week_number_in_year",
            "week_number_in_month",
            "month_number_in_year",
            "year_number",
        ],
    ),
)
def test_interface_correct_args(true_params: List[str], train_df: pd.DataFrame):
    """This test checks that feature generates all the expected columns and no unexpected ones in transform"""
    init_params = deepcopy(INIT_PARAMS_TEMPLATE)
    test_segs = train_df.columns.get_level_values(0).unique()
    for key in true_params:
        init_params[key] = True
    transform = DateFlagsTransform(**init_params)
    result = transform.fit_transform(df=train_df.copy())

    assert sorted(test_segs) == sorted(result.columns.get_level_values(0).unique())
    assert sorted(result.columns.names) == ["feature", "segment"]

    for seg in result.columns.get_level_values(0).unique():
        tmp_df = result[seg]
        assert sorted(list(tmp_df.columns)) == sorted(true_params + ["target"])
        for param in true_params:
            assert tmp_df[param].dtype == "category"


@pytest.mark.parametrize(
    "true_params",
    (["special_days_in_week"], ["special_days_in_month"], ["special_days_in_week", "special_days_in_month"]),
)
def test_interface_correct_tuple_args(true_params: List[str], train_df: pd.DataFrame):
    """This test checks that feature generates all the expected columns and no unexpected ones in transform"""
    init_params = deepcopy(INIT_PARAMS_TEMPLATE)
    test_segs = train_df.columns.get_level_values(0).unique()
    for key in true_params:
        init_params[key] = SPECIAL_DAYS
    transform = DateFlagsTransform(**init_params)
    result = transform.fit_transform(df=train_df.copy())

    assert sorted(test_segs) == sorted(result.columns.get_level_values(0).unique())
    assert sorted(result.columns.names) == ["feature", "segment"]

    for seg in result.columns.get_level_values(0).unique():
        tmp_df = result[seg]
        assert sorted(list(tmp_df.columns)) == sorted(true_params + ["target"])
        for param in true_params:
            assert tmp_df[param].dtype == "category"


@pytest.mark.parametrize(
    "true_params",
    (
        {"day_number_in_week": True},
        {"day_number_in_month": True},
        {"week_number_in_year": True},
        {"week_number_in_month": True},
        {"month_number_in_year": True},
        {"year_number": True},
        {"special_days_in_week": SPECIAL_DAYS},
        {"special_days_in_month": SPECIAL_DAYS},
    ),
)
def test_feature_values(
    true_params: Dict[str, Union[bool, Tuple[int, int]]], train_df: pd.DataFrame, dateflags_true_df: pd.DataFrame
):
    """This test checks that feature generates correct values"""
    init_params = deepcopy(INIT_PARAMS_TEMPLATE)
    init_params.update(true_params)
    transform = DateFlagsTransform(**init_params)
    result = transform.fit_transform(df=train_df.copy())

    segments_true = dateflags_true_df.columns.get_level_values(0).unique()
    segment_result = result.columns.get_level_values(0).unique()

    assert sorted(segment_result) == sorted(segments_true)

    for seg in segment_result:
        segment_true = dateflags_true_df[seg]
        true_df = segment_true[list(true_params.keys()) + ["target"]].sort_index(axis=1)
        result_df = result[seg].sort_index(axis=1)
        assert (true_df == result_df).all().all()
