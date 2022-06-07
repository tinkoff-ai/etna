import numpy as np
import pandas as pd
import pytest

from etna.datasets import TSDataset
from etna.datasets import generate_ar_df
from etna.datasets import generate_periodic_df
from etna.transforms.feature_selection import FilterFeaturesTransform


@pytest.fixture
def ts_with_features() -> TSDataset:
    timestamp = pd.date_range("2020-01-01", periods=100, freq="D")
    df_1 = pd.DataFrame({"timestamp": timestamp, "segment": "segment_1", "target": 1})
    df_2 = pd.DataFrame({"timestamp": timestamp, "segment": "segment_2", "target": 2})
    df = TSDataset.to_dataset(pd.concat([df_1, df_2], ignore_index=False))

    df_exog_1 = pd.DataFrame({"timestamp": timestamp, "segment": "segment_1", "exog_1": 1, "exog_2": 2})
    df_exog_2 = pd.DataFrame({"timestamp": timestamp, "segment": "segment_2", "exog_1": 3, "exog_2": 4})
    df_exog = TSDataset.to_dataset(pd.concat([df_exog_1, df_exog_2], ignore_index=False))

    return TSDataset(df=df, df_exog=df_exog, freq="D")


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


def test_set_only_include():
    """Test that transform is created with include."""
    _ = FilterFeaturesTransform(include=["exog_1", "exog_2"])


def test_set_only_exclude():
    """Test that transform is created with exclude."""
    _ = FilterFeaturesTransform(exclude=["exog_1", "exog_2"])


def test_set_include_and_exclude():
    """Test that transform is not created with include and exclude."""
    with pytest.raises(ValueError, match="There should be exactly one option set: include or exclude"):
        _ = FilterFeaturesTransform(include=["exog_1"], exclude=["exog_2"])


def test_set_none():
    """Test that transform is not created without include and exclude."""
    with pytest.raises(ValueError, match="There should be exactly one option set: include or exclude"):
        _ = FilterFeaturesTransform()


@pytest.mark.parametrize("include", [[], ["target"], ["exog_1"], ["exog_1", "exog_2", "target"]])
def test_include_filter(ts_with_features, include):
    """Test that transform remains only features in include."""
    original_df = ts_with_features.to_pandas()
    transform = FilterFeaturesTransform(include=include)
    ts_with_features.fit_transform([transform])
    df_transformed = ts_with_features.to_pandas()
    expected_columns = set(include)
    got_columns = set(df_transformed.columns.get_level_values("feature"))
    assert got_columns == expected_columns
    segments = ts_with_features.segments
    for column in got_columns:
        assert np.all(
            df_transformed.loc[:, pd.IndexSlice[segments, column]]
            == original_df.loc[:, pd.IndexSlice[segments, column]]
        )


@pytest.mark.parametrize(
    "exclude, expected_columns",
    [
        ([], ["target", "exog_1", "exog_2"]),
        (["target"], ["exog_1", "exog_2"]),
        (["exog_1", "exog_2"], ["target"]),
        (["target", "exog_1", "exog_2"], []),
    ],
)
def test_exclude_filter(ts_with_features, exclude, expected_columns):
    """Test that transform removes only features in exclude."""
    original_df = ts_with_features.to_pandas()
    transform = FilterFeaturesTransform(exclude=exclude)
    ts_with_features.fit_transform([transform])
    df_transformed = ts_with_features.to_pandas()
    got_columns = set(df_transformed.columns.get_level_values("feature"))
    assert got_columns == set(expected_columns)
    segments = ts_with_features.segments
    for column in got_columns:
        assert np.all(
            df_transformed.loc[:, pd.IndexSlice[segments, column]]
            == original_df.loc[:, pd.IndexSlice[segments, column]]
        )


def test_include_filter_wrong_column(ts_with_features):
    """Test that transform raises error with non-existent column in include."""
    transform = FilterFeaturesTransform(include=["non-existent-column"])
    with pytest.raises(ValueError, match="Features {.*} are not present in the dataset"):
        ts_with_features.fit_transform([transform])


def test_exclude_filter_wrong_column(ts_with_features):
    """Test that transform raises error with non-existent column in exclude."""
    transform = FilterFeaturesTransform(exclude=["non-existent-column"])
    with pytest.raises(ValueError, match="Features {.*} are not present in the dataset"):
        ts_with_features.fit_transform([transform])


@pytest.mark.parametrize("type", ("include", "exclude"))
@pytest.mark.parametrize(
    "columns, expected_excluded_columns, expected_included_columns",
    [
        ([], ["target", "exog_1", "exog_2"], []),
        (["target"], ["exog_1", "exog_2"], ["target"]),
        (["exog_1", "exog_2"], ["target"], ["exog_1", "exog_2"]),
        (["target", "exog_1", "exog_2"], [], ["target", "exog_1", "exog_2"]),
    ],
)
def test_inverse_transform_save_columns(
    ts_with_features, columns, expected_excluded_columns, expected_included_columns, type
):
    original_df = ts_with_features.to_pandas()
    transform = (
        FilterFeaturesTransform(exclude=columns, return_features=True)
        if type == "exclude"
        else FilterFeaturesTransform(include=columns, return_features=True)
    )
    ts_with_features.fit_transform([transform])
    df_transformed = transform._df_removed
    got_columns = set(df_transformed.columns.get_level_values("feature"))
    if type == "include":
        assert got_columns == set(expected_excluded_columns)
    else:
        assert got_columns == set(expected_included_columns)

    segments = ts_with_features.segments
    for column in got_columns:
        assert np.all(
            df_transformed.loc[:, pd.IndexSlice[segments, column]]
            == original_df.loc[:, pd.IndexSlice[segments, column]]
        )


@pytest.mark.parametrize("type", ("include", "exclude"))
@pytest.mark.parametrize(
    "columns, expected_excluded_columns, expected_included_columns",
    [
        ([], ["target", "exog_1", "exog_2"], []),
        (["target"], ["exog_1", "exog_2"], ["target"]),
        (["exog_1", "exog_2"], ["target"], ["exog_1", "exog_2"]),
        (["target", "exog_1", "exog_2"], [], ["target", "exog_1", "exog_2"]),
    ],
)
def test_inverse_transform_back_columns(
    ts_with_features, columns, expected_excluded_columns, expected_included_columns, type
):
    original_df = ts_with_features.to_pandas().copy()
    columns_original = set(original_df.columns)
    transform = (
        FilterFeaturesTransform(exclude=columns, return_features=True)
        if type == "exclude"
        else FilterFeaturesTransform(include=columns, return_features=True)
    )
    ts_with_features.fit_transform([transform])
    ts_with_features.inverse_transform()
    columns_inversed = set(ts_with_features.to_pandas().columns)
    assert columns_inversed == columns_original
    segments = ts_with_features.segments
    for column in columns_inversed:
        assert np.all(
            ts_with_features.to_pandas().loc[:, pd.IndexSlice[segments, column]]
            == original_df.loc[:, pd.IndexSlice[segments, column]]
        )
