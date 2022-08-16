from typing import List
from unittest.mock import Mock

import pandas as pd
import pytest

from etna.datasets import TSDataset
from etna.datasets import generate_ar_df
from etna.transforms import NewTransform
from etna.transforms.base import override


class NewTransformMock(NewTransform):
    def get_regressors_info(self) -> List[str]:
        return ["regressor_test"]

    def _fit(self, df: pd.DataFrame):
        pass

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    @override
    def _inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["target"] = -100
        return df


@pytest.fixture
def remove_columns_df():
    df = generate_ar_df(periods=10, n_segments=3, start_time="2000-01-01")
    df["exog_1"] = 1
    df = TSDataset.to_dataset(df)

    df_transformed = generate_ar_df(periods=10, n_segments=3, start_time="2000-01-01")
    df_transformed = TSDataset.to_dataset(df_transformed)
    return df, df_transformed


@pytest.mark.parametrize(
    "in_column, expected_features",
    [("all", "all"), ("target", ["target"]), (["target", "segment"], ["target", "segment"])],
)
def test_required_features(in_column, expected_features):
    transform = NewTransformMock(in_column=in_column)
    assert transform.required_features == expected_features


def test_update_dataset_remove_columns(remove_columns_df):
    ts = Mock()
    df, df_transformed = remove_columns_df
    expected_features_to_remove = list(
        set(df.columns.get_level_values("feature")) - set(df_transformed.columns.get_level_values("feature"))
    )
    transform = NewTransformMock()

    transform._update_dataset(ts=ts, df=df, df_transformed=df_transformed)
    ts.drop_features.assert_called_with(features=expected_features_to_remove, drop_from_exog=False)


def test_update_dataset_update_columns(remove_columns_df):
    ts = Mock()
    df_transformed, df = remove_columns_df
    transform = NewTransformMock()

    transform._update_dataset(ts=ts, df=df, df_transformed=df_transformed)
    ts.update_columns_from_pandas.assert_called_with(
        df_update=df_transformed, update_exog=False, regressors=transform.get_regressors_info()
    )


@pytest.mark.parametrize(
    "in_column, required_features",
    [("all", "all"), ("target", ["target"]), (["target", "segment"], ["target", "segment"])],
)
def test_fit_request_correct_columns(in_column, required_features):
    ts = Mock()
    transform = NewTransformMock(in_column=in_column)

    transform.fit(ts=ts)
    ts.to_pandas.assert_called_with(flatten=False, features=required_features)


@pytest.mark.parametrize(
    "in_column, required_features",
    [("all", "all"), ("target", ["target"]), (["target", "segment"], ["target", "segment"])],
)
def test_transform_request_correct_columns(in_column, required_features):
    ts = Mock()
    transform = NewTransformMock(in_column=in_column)
    transform._update_dataset = Mock()

    transform.transform(ts=ts)
    ts.to_pandas.assert_called_with(flatten=False, features=required_features)


@pytest.mark.parametrize(
    "in_column, required_features",
    [("all", "all"), ("target", ["target"]), (["target", "segment"], ["target", "segment"])],
)
def test_transform_request_update_dataset(remove_columns_df, in_column, required_features):
    df, _ = remove_columns_df
    ts = TSDataset(df=df, freq="D")
    ts.to_pandas = Mock(return_value=df)

    transform = NewTransformMock(in_column=in_column)
    transform._update_dataset = Mock()

    transform.transform(ts=ts)
    transform._update_dataset.assert_called_with(ts=ts, df=df, df_transformed=df)


def test_inverse_transform_update_dataset(remove_columns_df):
    df, _ = remove_columns_df
    ts = TSDataset(df=df, freq="D")

    transform = NewTransformMock()
    transform._update_dataset = Mock()

    transform.inverse_transform(ts=ts)
    transform._update_dataset.assert_called()


def test_inverse_transform_not_update_dataset_if_not_overriden(remove_columns_df):
    df, _ = remove_columns_df
    ts = TSDataset(df=df, freq="D")

    transform = NewTransformMock()
    transform._update_dataset = Mock()
    transform._inverse_transform = lambda df: df

    transform.inverse_transform(ts=ts)
    assert not transform._update_dataset.called
