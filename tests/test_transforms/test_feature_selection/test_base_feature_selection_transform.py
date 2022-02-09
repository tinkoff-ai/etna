import pandas as pd
import pytest

from etna.analysis import StatisticsRelevanceTable
from etna.datasets import TSDataset
from etna.datasets import generate_ar_df
from etna.transforms.feature_selection import MRMRFeatureSelectionTransform


@pytest.fixture
def df_with_complex_exog(random_seed) -> pd.DataFrame:
    df = generate_ar_df(periods=100, start_time="2020-01-01", n_segments=4)

    df_exog_1 = generate_ar_df(periods=100, start_time="2020-01-01", n_segments=4, random_seed=2).rename(
        {"target": "exog"}, axis=1
    )
    df_exog_2 = generate_ar_df(periods=150, start_time="2019-12-01", n_segments=4, random_seed=3).rename(
        {"target": "regressor_1"}, axis=1
    )
    df_exog_3 = generate_ar_df(periods=150, start_time="2019-12-01", n_segments=4, random_seed=4).rename(
        {"target": "regressor_2"}, axis=1
    )

    df_exog = pd.merge(df_exog_1, df_exog_2, on=["timestamp", "segment"], how="right")
    df_exog = pd.merge(df_exog, df_exog_3, on=["timestamp", "segment"])

    df = TSDataset.to_dataset(df)
    df_exog = TSDataset.to_dataset(df_exog)
    ts = TSDataset(df=df, freq="D", df_exog=df_exog, known_future=["regressor_1", "regressor_2"])
    return ts.df


@pytest.mark.parametrize(
    "features_to_use, expected_features",
    (
        ("all", ["regressor_1", "regressor_2", "exog"]),
        (["regressor_1"], ["regressor_1"]),
        (["regressor_1", "unknown_column"], ["regressor_1"]),
    ),
)
def test_get_features_to_use(df_with_complex_exog: pd.DataFrame, features_to_use, expected_features):
    base_selector = MRMRFeatureSelectionTransform(
        relevance_table=StatisticsRelevanceTable(), top_k=3, features_to_use=features_to_use
    )
    features = base_selector._get_features_to_use(df_with_complex_exog)
    assert sorted(features) == sorted(expected_features)


def test_get_features_to_use_raise_warning(df_with_complex_exog: pd.DataFrame):
    base_selector = MRMRFeatureSelectionTransform(
        relevance_table=StatisticsRelevanceTable(), top_k=3, features_to_use=["regressor_1", "unknown_column"]
    )
    with pytest.warns(
        UserWarning, match="Columns from feature_to_use which are out of dataframe columns will be dropped!"
    ):
        _ = base_selector._get_features_to_use(df_with_complex_exog)
