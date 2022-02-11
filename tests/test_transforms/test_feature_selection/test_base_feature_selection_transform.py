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
def test_get_features_to_use(df_with_exog: pd.DataFrame, features_to_use, expected_features):
    base_selector = MRMRFeatureSelectionTransform(
        relevance_table=StatisticsRelevanceTable(), top_k=3, features_to_use=features_to_use
    )
    features = base_selector._get_features_to_use(df_with_exog)
    assert sorted(features) == sorted(expected_features)


def test_get_features_to_use_raise_warning(df_with_exog: pd.DataFrame):
    base_selector = MRMRFeatureSelectionTransform(
        relevance_table=StatisticsRelevanceTable(), top_k=3, features_to_use=["regressor_1", "unknown_column"]
    )
    with pytest.warns(
        UserWarning, match="Columns from feature_to_use which are out of dataframe columns will be dropped!"
    ):
        _ = base_selector._get_features_to_use(df_with_exog)


@pytest.mark.parametrize(
    "features_to_use, selected_regressors, expected_columns",
    (
        ("all", ["regressor_1"], ["regressor_1", "target"]),
        (["regressor_1", "regressor_2"], ["regressor_1"], ["regressor_1", "exog", "target"]),
    ),
)
def test_transform(df_with_exog: pd.DataFrame, features_to_use, selected_regressors, expected_columns):
    base_selector = MRMRFeatureSelectionTransform(
        relevance_table=StatisticsRelevanceTable(), top_k=3, features_to_use=features_to_use
    )
    base_selector.selected_regressors = selected_regressors
    df_with_complex_exog = base_selector.transform(df_with_exog)
    columns = set(df_with_complex_exog.columns.get_level_values("feature"))
    assert sorted(columns) == sorted(expected_columns)
