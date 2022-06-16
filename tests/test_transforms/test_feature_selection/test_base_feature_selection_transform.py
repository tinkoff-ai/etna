import numpy as np
import pandas as pd
import pytest

from etna.analysis import StatisticsRelevanceTable
from etna.transforms.feature_selection import MRMRFeatureSelectionTransform


@pytest.mark.parametrize(
    "features_to_use, expected_features",
    (
        ("all", ["regressor_1", "regressor_2", "exog"]),
        (["regressor_1"], ["regressor_1"]),
        (["regressor_1", "unknown_column"], ["regressor_1"]),
    ),
)
def test_get_features_to_use(ts_with_exog: pd.DataFrame, features_to_use, expected_features):
    base_selector = MRMRFeatureSelectionTransform(
        relevance_table=StatisticsRelevanceTable(), top_k=3, features_to_use=features_to_use
    )
    features = base_selector._get_features_to_use(ts_with_exog.df)
    assert sorted(features) == sorted(expected_features)


def test_get_features_to_use_raise_warning(ts_with_exog: pd.DataFrame):
    base_selector = MRMRFeatureSelectionTransform(
        relevance_table=StatisticsRelevanceTable(), top_k=3, features_to_use=["regressor_1", "unknown_column"]
    )
    with pytest.warns(
        UserWarning, match="Columns from feature_to_use which are out of dataframe columns will be dropped!"
    ):
        _ = base_selector._get_features_to_use(ts_with_exog.df)


@pytest.mark.parametrize(
    "features_to_use, selected_features, expected_columns",
    (
        ("all", ["regressor_1"], ["regressor_1", "target"]),
        (["regressor_1", "regressor_2"], ["regressor_1"], ["regressor_1", "exog", "target"]),
    ),
)
def test_transform(ts_with_exog: pd.DataFrame, features_to_use, selected_features, expected_columns):
    base_selector = MRMRFeatureSelectionTransform(
        relevance_table=StatisticsRelevanceTable(),
        top_k=3,
        features_to_use=features_to_use,
        return_features=False,
    )
    base_selector.selected_features = selected_features
    transformed_df_with_exog = base_selector.transform(ts_with_exog.df)
    columns = set(transformed_df_with_exog.columns.get_level_values("feature"))
    assert sorted(columns) == sorted(expected_columns)


@pytest.mark.parametrize("return_features", [True, False])
@pytest.mark.parametrize(
    "features_to_use, selected_features, expected_columns",
    (
        ("all", ["regressor_1"], ["exog", "regressor_2"]),
        (["regressor_1", "regressor_2"], ["regressor_1"], ["regressor_2"]),
    ),
)
def test_transform_save_columns(ts_with_exog, features_to_use, selected_features, expected_columns, return_features):
    original_df = ts_with_exog.to_pandas()
    transform = MRMRFeatureSelectionTransform(
        relevance_table=StatisticsRelevanceTable(),
        top_k=3,
        features_to_use=features_to_use,
        return_features=return_features,
    )
    transform.selected_features = selected_features
    ts_with_exog.transform([transform])
    df_saved = transform._df_removed
    if return_features:
        got_columns = set(df_saved.columns.get_level_values("feature"))
        assert got_columns == set(expected_columns)
        for column in got_columns:
            assert np.all(df_saved.loc[:, pd.IndexSlice[:, column]] == original_df.loc[:, pd.IndexSlice[:, column]])
    else:
        assert df_saved is None


@pytest.mark.parametrize(
    "features_to_use, expected_columns, return_features",
    [
        ("all", ["exog", "regressor_1", "regressor_2", "target"], True),
        (["regressor_1", "regressor_2"], ["regressor_2", "regressor_1", "exog", "target"], False),
        ("all", ["regressor_2", "exog", "target"], False),
        (["regressor_1", "regressor_2"], ["regressor_2", "regressor_1", "exog", "target"], True),
    ],
)
def test_inverse_transform_back_excluded_columns(ts_with_exog, features_to_use, return_features, expected_columns):
    original_df = ts_with_exog.to_pandas()
    transform = MRMRFeatureSelectionTransform(
        relevance_table=StatisticsRelevanceTable(),
        top_k=2,
        features_to_use=features_to_use,
        return_features=return_features,
    )
    ts_with_exog.fit_transform([transform])
    ts_with_exog.inverse_transform()
    columns_inversed = set(ts_with_exog.columns.get_level_values("feature"))
    assert columns_inversed == set(expected_columns)
    for column in columns_inversed:
        assert np.all(ts_with_exog[:, :, column] == original_df.loc[:, pd.IndexSlice[:, column]])
