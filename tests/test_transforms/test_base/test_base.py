from typing import List
from unittest.mock import Mock

import pandas as pd
import pytest

from etna.datasets import TSDataset
from etna.datasets import generate_ar_df
from etna.transforms import IrreversibleTransform
from etna.transforms import ReversibleTransform


class TransformMock(IrreversibleTransform):
    def get_regressors_info(self) -> List[str]:
        return ["regressor_test"]

    def _fit(self, df: pd.DataFrame):
        pass

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df


class ReversibleTransformMock(ReversibleTransform):
    def get_regressors_info(self) -> List[str]:
        return ["regressor_test"]

    def _fit(self, df: pd.DataFrame):
        pass

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def _inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df


@pytest.fixture
def remove_columns_df():
    df = generate_ar_df(periods=10, n_segments=3, start_time="2000-01-01")
    df["exog_1"] = 1
    df["target_0.01"] = 2
    df = TSDataset.to_dataset(df)

    df_transformed = generate_ar_df(periods=10, n_segments=3, start_time="2000-01-01")
    df_transformed = TSDataset.to_dataset(df_transformed)
    return df, df_transformed


@pytest.mark.parametrize(
    "required_features, expected_features",
    [("all", "all"), (["target", "segment"], ["target", "segment"])],
)
def test_required_features(required_features, expected_features):
    transform = TransformMock(required_features=required_features)
    assert transform.required_features == expected_features


def test_update_dataset_remove_columns(remove_columns_df):
    df, df_transformed = remove_columns_df
    columns_before = set(df.columns.get_level_values("feature"))
    ts = TSDataset(df=df, freq="D")
    ts.drop_features = Mock()
    expected_features_to_remove = list(
        set(df.columns.get_level_values("feature")) - set(df_transformed.columns.get_level_values("feature"))
    )
    transform = TransformMock(required_features=["target"])

    transform._update_dataset(ts=ts, columns_before=columns_before, df_transformed=df_transformed)
    ts.drop_features.assert_called_with(features=expected_features_to_remove, drop_from_exog=False)


def test_update_dataset_update_columns(remove_columns_df):
    df, df_transformed = remove_columns_df
    columns_before = set(df.columns.get_level_values("feature"))
    ts = TSDataset(df=df, freq="D")
    ts.update_columns_from_pandas = Mock()
    transform = TransformMock(required_features=["target"])

    transform._update_dataset(ts=ts, columns_before=columns_before, df_transformed=df_transformed)
    ts.update_columns_from_pandas.assert_called()


def test_update_dataset_add_columns(remove_columns_df):
    df_transformed, df = remove_columns_df
    columns_before = set(df.columns.get_level_values("feature"))
    ts = TSDataset(df=df, freq="D")
    ts.add_columns_from_pandas = Mock()
    transform = TransformMock(required_features=["target"])

    transform._update_dataset(ts=ts, columns_before=columns_before, df_transformed=df_transformed)
    ts.add_columns_from_pandas.assert_called()


@pytest.mark.parametrize(
    "required_features",
    [("all"), (["target", "segment"])],
)
def test_fit_request_correct_columns(required_features):
    ts = Mock()
    transform = TransformMock(required_features=required_features)

    transform.fit(ts=ts)
    ts.to_pandas.assert_called_with(flatten=False, features=required_features)


@pytest.mark.parametrize(
    "required_features",
    [("all"), (["target", "segment"])],
)
def test_transform_request_correct_columns(remove_columns_df, required_features):
    df, _ = remove_columns_df
    ts = TSDataset(df=df, freq="D")
    ts.to_pandas = Mock(return_value=df)

    transform = TransformMock(required_features=required_features)
    transform._update_dataset = Mock()

    transform.transform(ts=ts)
    ts.to_pandas.assert_called_with(flatten=False, features=required_features)


@pytest.mark.parametrize(
    "required_features",
    [("all"), (["target", "segment"])],
)
def test_transform_request_update_dataset(remove_columns_df, required_features):
    df, _ = remove_columns_df
    columns_before = set(df.columns.get_level_values("feature"))
    ts = TSDataset(df=df, freq="D")
    ts.to_pandas = Mock(return_value=df)

    transform = TransformMock(required_features=required_features)
    transform._update_dataset = Mock()

    transform.transform(ts=ts)
    transform._update_dataset.assert_called_with(ts=ts, columns_before=columns_before, df_transformed=df)


@pytest.mark.parametrize(
    "in_column, expected_required_features",
    [(["target"], ["target", "target_0.01"]), (["exog_1"], ["exog_1"])],
)
def test_inverse_transform_add_target_quantiles(remove_columns_df, in_column, expected_required_features):
    df, _ = remove_columns_df
    ts = TSDataset(df=df, freq="D")

    transform = ReversibleTransformMock(required_features=in_column)
    required_features = transform._get_inverse_transform_required_features(ts)
    assert sorted(required_features) == sorted(expected_required_features)


def test_inverse_transform_request_update_dataset(remove_columns_df):
    df, _ = remove_columns_df
    columns_before = set(df.columns.get_level_values("feature"))
    ts = TSDataset(df=df, freq="D")
    ts.to_pandas = Mock(return_value=df)

    transform = ReversibleTransformMock(required_features="all")
    transform._inverse_transform = Mock()
    transform._update_dataset = Mock()

    transform.inverse_transform(ts=ts)
    expected_df_transformed = transform._inverse_transform.return_value
    transform._update_dataset.assert_called_with(
        ts=ts, columns_before=columns_before, df_transformed=expected_df_transformed
    )
