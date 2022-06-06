from copy import copy

import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeRegressor

from etna.analysis import StatisticsRelevanceTable
from etna.datasets import TSDataset
from etna.metrics import MAE
from etna.models import ProphetModel
from etna.pipeline import Pipeline
from etna.transforms import LogTransform
from etna.transforms.feature_selection import FilterFeaturesTransform
from etna.transforms.feature_selection.feature_importance import MRMRFeatureSelectionTransform
from etna.transforms.feature_selection.feature_importance import TreeFeatureSelectionTransform


@pytest.fixture
def ts_with_features() -> TSDataset:
    timestamp = pd.date_range("2020-01-01", periods=100, freq="D")
    df_1 = pd.DataFrame({"timestamp": timestamp, "segment": "segment_1", "target": 1.0})
    df_2 = pd.DataFrame({"timestamp": timestamp, "segment": "segment_2", "target": 2.0})
    df = TSDataset.to_dataset(pd.concat([df_1, df_2], ignore_index=False))

    df_exog_1 = pd.DataFrame({"timestamp": timestamp, "segment": "segment_1", "exog_1": 1.0, "exog_2": 2.0})
    df_exog_2 = pd.DataFrame({"timestamp": timestamp, "segment": "segment_2", "exog_1": 3.0, "exog_2": 4.0})
    df_exog = TSDataset.to_dataset(pd.concat([df_exog_1, df_exog_2], ignore_index=False))

    return TSDataset(df=df, df_exog=df_exog, freq="D")


@pytest.fixture()
def sinusoid_ts():
    periods = 1000
    sinusoid_ts_1 = pd.DataFrame(
        {
            "segment": np.zeros(periods),
            "timestamp": pd.date_range(start="1/1/2018", periods=periods),
            "target": [np.sin(i / 10) + i / 5 for i in range(periods)],
            "feature_1": [i / 10 for i in range(periods)],
            "feature_2": [np.sin(i) for i in range(periods)],
            "feature_3": [np.cos(i / 10) for i in range(periods)],
            "feature_4": [np.cos(i) for i in range(periods)],
            "feature_5": [i ** 2 for i in range(periods)],
            "feature_6": [i * np.sin(i) for i in range(periods)],
            "feature_7": [i * np.cos(i) for i in range(periods)],
            "feature_8": [i + np.cos(i) for i in range(periods)],
        }
    )
    df = TSDataset.to_dataset(sinusoid_ts_1)
    ts = TSDataset(df, freq="D")
    return ts


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
def test_filter(ts_with_features, columns, expected_excluded_columns, expected_included_columns, type):
    original_df = ts_with_features.to_pandas()
    transform = (
        FilterFeaturesTransform(exclude=columns, return_features=True)
        if type == "exclude"
        else FilterFeaturesTransform(include=columns, return_features=True)
    )
    ts_with_features.fit_transform([transform])
    df_transformed = ts_with_features.to_pandas()
    got_columns = set(df_transformed.columns.get_level_values("feature"))
    if type == "include":
        assert got_columns == set(expected_included_columns)
    else:
        assert got_columns == set(expected_excluded_columns)

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
def test_transform_inverse(ts_with_features, columns, expected_excluded_columns, expected_included_columns, type):
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


@pytest.mark.parametrize("type", ("include", "exclude"))
@pytest.mark.parametrize(
    "expected_excluded_columns, expected_included_columns",
    [
        (["exog_1"], ["target", "exog_2"]),
    ],
)
def test_transform_inverse_transform(ts_with_features, expected_excluded_columns, expected_included_columns, type):
    transform = (
        FilterFeaturesTransform(exclude=expected_excluded_columns, return_features=True)
        if type == "exclude"
        else FilterFeaturesTransform(include=expected_included_columns, return_features=True)
    )
    transforms = [LogTransform(in_column="exog_1"), transform, LogTransform(in_column="exog_2")]

    ts_with_features.fit_transform(transforms)

    original_df = ts_with_features.to_pandas().copy()
    columns_original = set(original_df.columns)

    ts_with_features.inverse_transform()
    ts_with_features.transform(transforms)
    columns_inversed = set(ts_with_features.to_pandas().columns)
    assert columns_inversed == columns_original
    segments = ts_with_features.segments

    eps = 1e-9

    for column in columns_inversed:
        assert np.all(
            abs(
                ts_with_features.to_pandas().loc[:, pd.IndexSlice[segments, column]]
                - original_df.loc[:, pd.IndexSlice[segments, column]]
            )
            < eps
        )


@pytest.mark.parametrize(
    "model",
    [
        DecisionTreeRegressor(random_state=42),
        ExtraTreeRegressor(random_state=42),
        RandomForestRegressor(n_estimators=10, random_state=42),
        ExtraTreesRegressor(n_estimators=10, random_state=42),
        GradientBoostingRegressor(n_estimators=10, random_state=42),
        CatBoostRegressor(iterations=10, random_state=42, silent=True),
    ],
)
def test_tree_selector(ts_with_features, model):
    ts1 = ts_with_features
    ts2 = copy(ts1)
    original_df = ts1.to_pandas().copy()

    selector_return_features = TreeFeatureSelectionTransform(
        model=model, top_k=3, features_to_use="all", return_features=True
    )
    ts1.fit_transform([selector_return_features])
    selector = TreeFeatureSelectionTransform(model=model, top_k=3, features_to_use="all", return_features=False)
    ts2.fit_transform([selector])
    assert set(ts1.columns) == set(ts2.columns)
    ts1.inverse_transform()
    eps = 1e-9
    columns_inversed = set(ts1.to_pandas().columns)
    segments = ts_with_features.segments

    for column in columns_inversed:
        assert np.all(
            abs(
                ts_with_features.to_pandas().loc[:, pd.IndexSlice[segments, column]]
                - original_df.loc[:, pd.IndexSlice[segments, column]]
            )
            < eps
        )


@pytest.mark.parametrize("relevance_table", ([StatisticsRelevanceTable()]))
@pytest.mark.parametrize("top_k", [0, 1, 2, 5, 6])
def test_mrmr(relevance_table, top_k, sinusoid_ts):
    ts1 = sinusoid_ts
    ts2 = copy(ts1)
    original_df = ts1.to_pandas().copy()

    selector_return_features = MRMRFeatureSelectionTransform(
        relevance_table=relevance_table, top_k=top_k, return_features=True
    )
    ts1.fit_transform([selector_return_features])

    selector = MRMRFeatureSelectionTransform(relevance_table=relevance_table, top_k=top_k)
    ts2.fit_transform([selector])
    assert set(ts1.columns) == set(ts2.columns)
    ts1.inverse_transform()
    eps = 1e-9
    columns_inversed = set(ts1.to_pandas().columns)
    segments = sinusoid_ts.segments

    for column in columns_inversed:
        assert np.all(
            abs(
                sinusoid_ts.to_pandas().loc[:, pd.IndexSlice[segments, column]]
                - original_df.loc[:, pd.IndexSlice[segments, column]]
            )
            < eps
        )


@pytest.mark.parametrize("relevance_table", ([StatisticsRelevanceTable()]))
@pytest.mark.parametrize(
    "model",
    [
        DecisionTreeRegressor(random_state=42),
    ],
)
def test_pipeline_backtest_filter(sinusoid_ts, model, relevance_table):
    ts = sinusoid_ts

    pipeline = Pipeline(
        model=ProphetModel(),
        transforms=[
            LogTransform(in_column="feature_1"),
            FilterFeaturesTransform(exclude=["feature_5"], return_features=True),
            TreeFeatureSelectionTransform(model=model, top_k=5, return_features=True),
            MRMRFeatureSelectionTransform(relevance_table=relevance_table, top_k=4, return_features=True),
        ],
        horizon=10,
    )
    metrics_df, forecast_df, fold_info_df = pipeline.backtest(ts=ts, metrics=[MAE()], aggregate_metrics=True)
    assert metrics_df["MAE"].iloc[0] < 2
