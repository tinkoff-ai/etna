from typing import Any
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

from etna.datasets import TSDataset
from etna.transforms import MaxAbsScalerTransform
from etna.transforms import MinMaxScalerTransform
from etna.transforms import RobustScalerTransform
from etna.transforms import StandardScalerTransform
from etna.transforms.sklearn import SklearnTransform
from etna.transforms.sklearn import TransformMode


class DummySkTransform:
    def fit(self, X, y=None):  # noqa: N803
        pass

    def transform(self, X, y=None):  # noqa: N803
        return X

    def inverse_transform(self, X, y=None):  # noqa: N803
        return X


class DummyTransform(SklearnTransform):
    def __init__(
        self,
        in_column: Optional[Union[str, List[str]]] = None,
        inplace: bool = True,
        out_column: str = None,
        mode: Union[TransformMode, str] = "per-segment",
    ):
        self.in_column = in_column
        self.inplace = inplace
        self.out_column = out_column
        self.mode = TransformMode(mode)
        super().__init__(
            in_column=in_column,
            inplace=inplace,
            out_column=out_column if out_column is not None else self.__repr__(),
            transformer=DummySkTransform(),
            mode=mode,
        )


@pytest.fixture
def normal_distributed_df() -> pd.DataFrame:
    df_1 = pd.DataFrame.from_dict({"timestamp": pd.date_range("2021-06-01", "2021-07-01", freq="1d")})
    df_2 = pd.DataFrame.from_dict({"timestamp": pd.date_range("2021-06-01", "2021-07-01", freq="1d")})
    generator = np.random.RandomState(seed=1)
    df_1["segment"] = "Moscow"
    df_1["target"] = generator.normal(loc=0, scale=10, size=len(df_1))
    df_1["exog"] = generator.normal(loc=2, scale=10, size=len(df_1))
    df_2["segment"] = "Omsk"
    df_2["target"] = generator.normal(loc=5, scale=1, size=len(df_2))
    df_2["exog"] = generator.normal(loc=3, scale=1, size=len(df_2))
    classic_df = pd.concat([df_1, df_2], ignore_index=True)
    return TSDataset.to_dataset(classic_df)


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
def test_transform_invalid_mode(scaler):
    """Check scaler behavior in case of invalid transform mode"""
    with pytest.raises(ValueError):
        _ = scaler(mode="a")


@pytest.mark.parametrize(
    "scaler",
    (
        DummyTransform(),
        StandardScalerTransform(),
        RobustScalerTransform(),
        MinMaxScalerTransform(),
        MaxAbsScalerTransform(),
        StandardScalerTransform(with_std=False),
        RobustScalerTransform(with_centering=False, with_scaling=False),
        MinMaxScalerTransform(feature_range=(5, 10)),
    ),
)
@pytest.mark.parametrize("mode", ("macro", "per-segment"))
def test_dummy_inverse_transform_all_columns(normal_distributed_df, scaler, mode):
    """Check that `inverse_transform(transform(df)) == df` for all columns."""
    scaler.mode = TransformMode(mode)
    feature_df = scaler.fit_transform(df=normal_distributed_df.copy())
    inversed_df = scaler.inverse_transform(df=feature_df.copy())
    npt.assert_array_almost_equal(normal_distributed_df.values, inversed_df.values)


@pytest.mark.parametrize(
    "scaler",
    (
        DummyTransform(in_column="target"),
        StandardScalerTransform(in_column="target"),
        RobustScalerTransform(in_column="target"),
        MinMaxScalerTransform(in_column="target"),
        MaxAbsScalerTransform(in_column="target"),
        StandardScalerTransform(in_column="target", with_std=False),
        RobustScalerTransform(in_column="target", with_centering=False, with_scaling=False),
        MinMaxScalerTransform(in_column="target", feature_range=(5, 10)),
    ),
)
@pytest.mark.parametrize("mode", ("macro", "per-segment"))
def test_dummy_inverse_transform_one_column(normal_distributed_df, scaler, mode):
    """Check that `inverse_transform(transform(df)) == df` for one column."""
    scaler.mode = TransformMode(mode)
    feature_df = scaler.fit_transform(df=normal_distributed_df.copy())
    inversed_df = scaler.inverse_transform(df=feature_df)
    npt.assert_array_almost_equal(normal_distributed_df.values, inversed_df.values)


@pytest.mark.parametrize(
    "scaler",
    (
        DummyTransform,
        StandardScalerTransform,
        RobustScalerTransform,
        MinMaxScalerTransform,
        MaxAbsScalerTransform,
        StandardScalerTransform,
        RobustScalerTransform,
        MinMaxScalerTransform,
    ),
)
@pytest.mark.parametrize("mode", ("macro", "per-segment"))
def test_inverse_transform_not_inplace(normal_distributed_df, scaler, mode):
    """Check that inversed values the same for not inplace version."""
    inplace_scaler = scaler(mode=mode)
    not_inplace_scaler = scaler(inplace=False, mode=mode)
    columns_to_compare = pd.MultiIndex.from_tuples(
        [(segment_name, not_inplace_scaler.__repr__()) for segment_name, _ in normal_distributed_df.columns]
    )
    inplace_feature_df = inplace_scaler.fit_transform(df=normal_distributed_df.copy())
    not_inplace_feature_df = not_inplace_scaler.fit_transform(df=normal_distributed_df.copy())

    inplace_feature_df.columns = columns_to_compare
    npt.assert_array_almost_equal(
        inplace_feature_df.loc[:, columns_to_compare].values, not_inplace_feature_df.loc[:, columns_to_compare]
    )


@pytest.mark.parametrize(
    "scaler",
    (
        DummyTransform,
        StandardScalerTransform,
        RobustScalerTransform,
        MinMaxScalerTransform,
        MaxAbsScalerTransform,
        StandardScalerTransform,
        RobustScalerTransform,
        MinMaxScalerTransform,
    ),
)
@pytest.mark.parametrize("mode", ("macro", "per-segment"))
def test_interface_out_column(normal_distributed_df: pd.DataFrame, scaler: Any, mode: str):
    """Check transform interface in non inplace mode with given out_column param."""
    out_column = "result"
    transform = scaler(inplace=False, mode=mode, out_column=out_column)
    result = transform.fit_transform(df=normal_distributed_df)
    for segment in result.columns.get_level_values("segment").unique():
        assert out_column in result[segment].columns


@pytest.mark.parametrize(
    "scaler",
    (
        DummyTransform,
        StandardScalerTransform,
        RobustScalerTransform,
        MinMaxScalerTransform,
        MaxAbsScalerTransform,
        StandardScalerTransform,
        RobustScalerTransform,
        MinMaxScalerTransform,
    ),
)
@pytest.mark.parametrize("mode", ("macro", "per-segment"))
def test_interface_repr(normal_distributed_df: pd.DataFrame, scaler: Any, mode: str):
    """Check transform interface in non inplace mode without given out_column param."""
    transform = scaler(inplace=False, mode=mode)
    excepted_column = transform.__repr__()
    result = transform.fit_transform(df=normal_distributed_df)
    for segment in result.columns.get_level_values("segment").unique():
        assert excepted_column in result[segment].columns
