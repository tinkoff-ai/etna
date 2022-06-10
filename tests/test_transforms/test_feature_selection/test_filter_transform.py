import numpy as np
import pandas as pd
import pytest

from etna.datasets import TSDataset
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
    for column in got_columns:
        assert np.all(df_transformed.loc[:, pd.IndexSlice[:, column]] == original_df.loc[:, pd.IndexSlice[:, column]])


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
    for column in got_columns:
        assert np.all(df_transformed.loc[:, pd.IndexSlice[:, column]] == original_df.loc[:, pd.IndexSlice[:, column]])


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


@pytest.mark.parametrize("return_features", [True, False])
@pytest.mark.parametrize(
    "columns, saved_columns",
    [
        ([], []),
        (["target"], ["target"]),
        (["exog_1", "exog_2"], ["exog_1", "exog_2"]),
        (["target", "exog_1", "exog_2"], ["target", "exog_1", "exog_2"]),
    ],
)
def test_transform_exclude_save_columns(ts_with_features, columns, saved_columns, return_features):
    original_df = ts_with_features.to_pandas()
    transform = FilterFeaturesTransform(exclude=columns, return_features=return_features)
    ts_with_features.fit_transform([transform])
    df_transformed = transform._df_removed
    if return_features:
        got_columns = set(df_transformed.columns.get_level_values("feature"))
        assert got_columns == set(saved_columns)
        for column in got_columns:
            assert np.all(
                df_transformed.loc[:, pd.IndexSlice[:, column]] == original_df.loc[:, pd.IndexSlice[:, column]]
            )
    else:
        assert df_transformed is None


@pytest.mark.parametrize("return_features", [True, False])
@pytest.mark.parametrize(
    "columns, saved_columns",
    [
        ([], ["target", "exog_1", "exog_2"]),
        (["target"], ["exog_1", "exog_2"]),
        (["exog_1", "exog_2"], ["target"]),
        (["target", "exog_1", "exog_2"], []),
    ],
)
def test_transform_include_save_columns(ts_with_features, columns, saved_columns, return_features):
    original_df = ts_with_features.to_pandas()
    transform = FilterFeaturesTransform(include=columns, return_features=return_features)
    ts_with_features.fit_transform([transform])
    df_transformed = transform._df_removed
    if return_features:
        got_columns = set(df_transformed.columns.get_level_values("feature"))
        assert got_columns == set(saved_columns)
        for column in got_columns:
            assert np.all(
                df_transformed.loc[:, pd.IndexSlice[:, column]] == original_df.loc[:, pd.IndexSlice[:, column]]
            )
    else:
        assert df_transformed is None


@pytest.mark.parametrize(
    "columns, return_features, expected_columns",
    [
        ([], True, ["exog_1", "target", "exog_2"]),
        ([], False, ["target", "exog_1", "exog_2"]),
        (["target"], True, ["exog_1", "target", "exog_2"]),
        (["target"], False, ["exog_2", "exog_1"]),
        (["exog_1", "exog_2"], True, ["exog_1", "target", "exog_2"]),
        (["exog_1", "exog_2"], False, ["target"]),
        (["target", "exog_1", "exog_2"], True, ["exog_1", "target", "exog_2"]),
        (["target", "exog_1", "exog_2"], False, []),
    ],
)
def test_inverse_transform_back_excluded_columns(ts_with_features, columns, return_features, expected_columns):
    original_df = ts_with_features.to_pandas()
    transform = FilterFeaturesTransform(exclude=columns, return_features=return_features)
    ts_with_features.fit_transform([transform])
    ts_with_features.inverse_transform()
    columns_inversed = set(ts_with_features.columns.get_level_values("feature"))
    assert columns_inversed == set(expected_columns)
    for column in ts_with_features.columns:
        assert np.all(ts_with_features[:, :, column] == original_df.loc[:, pd.IndexSlice[:, column]])


@pytest.mark.parametrize(
    "columns, return_features, expected_columns",
    [
        ([], True, ["exog_1", "target", "exog_2"]),
        ([], False, []),
        (["target"], True, ["exog_1", "target", "exog_2"]),
        (["target"], False, ["target"]),
        (["exog_1", "exog_2"], True, ["exog_1", "target", "exog_2"]),
        (["exog_1", "exog_2"], False, ["exog_1", "exog_2"]),
        (["target", "exog_1", "exog_2"], True, ["exog_1", "target", "exog_2"]),
        (["target", "exog_1", "exog_2"], False, ["exog_1", "target", "exog_2"]),
    ],
)
def test_inverse_transform_back_included_columns(ts_with_features, columns, return_features, expected_columns):
    original_df = ts_with_features.to_pandas()
    transform = FilterFeaturesTransform(include=columns, return_features=return_features)
    ts_with_features.fit_transform([transform])
    ts_with_features.inverse_transform()
    columns_inversed = set(ts_with_features.columns.get_level_values("feature"))
    assert columns_inversed == set(expected_columns)
    for column in ts_with_features.columns:
        assert np.all(ts_with_features[:, :, column] == original_df.loc[:, pd.IndexSlice[:, column]])
