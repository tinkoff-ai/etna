import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

from etna.datasets import TSDataset
from etna.transforms import MaxAbsScalerTransform
from etna.transforms import MinMaxScalerTransform
from etna.transforms import RobustScalerTransform
from etna.transforms import StandardScalerTransform


@pytest.fixture
def normal_distributed_df() -> pd.DataFrame:
    df_1 = pd.DataFrame.from_dict({"timestamp": pd.date_range("2021-06-01", "2021-07-01", freq="1d")})
    df_2 = pd.DataFrame.from_dict({"timestamp": pd.date_range("2021-06-01", "2021-07-01", freq="1d")})
    generator = np.random.RandomState(seed=1)
    df_1["segment"] = "Moscow"
    df_1["target"] = generator.normal(loc=0, scale=10, size=len(df_1))
    df_2["segment"] = "Omsk"
    df_2["target"] = generator.normal(loc=5, scale=1, size=len(df_1))
    classic_df = pd.concat([df_1, df_2], ignore_index=True)
    return TSDataset.to_dataset(classic_df)


@pytest.mark.parametrize(
    "scaler",
    (
        StandardScalerTransform(),
        RobustScalerTransform(),
        MinMaxScalerTransform(),
        MaxAbsScalerTransform(),
        StandardScalerTransform(with_std=False),
        RobustScalerTransform(with_centering=False, with_scaling=False),
        MinMaxScalerTransform(feature_range=(5, 10)),
    ),
)
def test_dummy_inverse_transform_all_columns(normal_distributed_df, scaler):
    """Check that `inverse_transform(transform(df)) == df` for all columns."""
    feature_df = scaler.fit_transform(df=normal_distributed_df.copy())
    inversed_df = scaler.inverse_transform(df=feature_df)
    npt.assert_array_almost_equal(normal_distributed_df.values, inversed_df.values)


@pytest.mark.parametrize(
    "scaler",
    (
        StandardScalerTransform(in_column="target"),
        RobustScalerTransform(in_column="target"),
        MinMaxScalerTransform(in_column="target"),
        MaxAbsScalerTransform(in_column="target"),
        StandardScalerTransform(in_column="target", with_std=False),
        RobustScalerTransform(in_column="target", with_centering=False, with_scaling=False),
        MinMaxScalerTransform(in_column="target", feature_range=(5, 10)),
    ),
)
def test_dummy_inverse_transform_one_column(normal_distributed_df, scaler):
    """Check that `inverse_transform(transform(df)) == df` for one column."""
    feature_df = scaler.fit_transform(df=normal_distributed_df.copy())
    inversed_df = scaler.inverse_transform(df=feature_df)
    npt.assert_array_almost_equal(normal_distributed_df.values, inversed_df.values)


@pytest.mark.parametrize(
    "scaler",
    (
        StandardScalerTransform,
        RobustScalerTransform,
        MinMaxScalerTransform,
        MaxAbsScalerTransform,
        StandardScalerTransform,
        RobustScalerTransform,
        MinMaxScalerTransform,
    ),
)
def test_dummy_inverse_transform_not_inplace(normal_distributed_df, scaler):
    """Check that inversed values the same for not inplace version."""
    inplace_scaler = scaler()
    not_inplace_scaler = scaler(inplace=False)
    inplace_feature_df = inplace_scaler.fit_transform(df=normal_distributed_df.copy())
    not_inplace_feature_df = not_inplace_scaler.fit_transform(df=normal_distributed_df.copy())
    columns_to_compare = pd.MultiIndex.from_tuples(
        [
            (segment_name, f"{str(inplace_scaler)}_{feature_name}")
            for segment_name, feature_name in normal_distributed_df.columns
        ]
    )

    inplace_feature_df.columns = columns_to_compare
    npt.assert_array_almost_equal(
        inplace_feature_df.loc[:, columns_to_compare].values, not_inplace_feature_df.loc[:, columns_to_compare]
    )
