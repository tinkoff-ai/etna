import numpy as np
import pytest

from etna.transforms import AddConstTransform
from etna.transforms import BoxCoxTransform
from etna.transforms import LogTransform
from etna.transforms import MaxAbsScalerTransform
from etna.transforms import MinMaxScalerTransform
from etna.transforms import RobustScalerTransform
from etna.transforms import StandardScalerTransform
from etna.transforms import YeoJohnsonTransform


def test_standard_scaler_dummy_mean_shift_for_quantiles_per_segment(toy_dataset_with_mean_shift_in_target):
    """
    This test checks that StandardScalerTransform.fit_transform + StandardScalerTransform.inverse_transform
    does not affect target's quantiles.
    """
    toy_dataset = toy_dataset_with_mean_shift_in_target
    scaler = StandardScalerTransform(in_column="target", with_std=False)
    toy_dataset = scaler.fit_transform(toy_dataset)
    toy_dataset = scaler.inverse_transform(toy_dataset)
    np.testing.assert_allclose(toy_dataset.iloc[:, 0], toy_dataset.iloc[:, 1])
    np.testing.assert_allclose(toy_dataset.iloc[:, 2], toy_dataset.iloc[:, 3])


def test_standard_scaler_dummy_mean_shift_for_quantiles_macro(toy_dataset_with_mean_shift_in_target):
    """This test checks that StandardScalerTransform.inverse_transform works correctly in macro mode."""
    toy_dataset = toy_dataset_with_mean_shift_in_target
    scaler = StandardScalerTransform(in_column="target", with_std=False, mode="macro")
    mean_1 = toy_dataset.iloc[:, 0].mean()
    mean_2 = toy_dataset.iloc[:, 2].mean()
    toy_dataset = scaler.fit_transform(toy_dataset)
    toy_dataset = scaler.inverse_transform(toy_dataset)
    np.testing.assert_allclose(toy_dataset.iloc[:, 0], toy_dataset.iloc[:, 1] - (mean_1 + mean_2) / 2 + mean_1)
    np.testing.assert_allclose(toy_dataset.iloc[:, 2], toy_dataset.iloc[:, 3] - (mean_1 + mean_2) / 2 + mean_2)


def test_add_constant_dummy(toy_dataset_equal_targets_and_quantiles):
    """
    This test checks that inverse_transform transforms forecast's quantiles the same way with target itself and
    transform does not affect quantiles.
    """
    toy_dataset = toy_dataset_equal_targets_and_quantiles
    shift = 10.0
    add_constant = AddConstTransform(in_column="target", value=shift)
    toy_dataset_transformed = add_constant.fit_transform(toy_dataset.copy())

    np.testing.assert_allclose(toy_dataset_transformed.iloc[:, 0] - shift, toy_dataset.iloc[:, 1])
    np.testing.assert_allclose(toy_dataset_transformed.iloc[:, 2] - shift, toy_dataset.iloc[:, 3])

    toy_dataset = add_constant.inverse_transform(toy_dataset)

    np.testing.assert_allclose(toy_dataset.iloc[:, 0], toy_dataset.iloc[:, 1])
    np.testing.assert_allclose(toy_dataset.iloc[:, 2], toy_dataset.iloc[:, 3])


@pytest.mark.parametrize(
    "transform",
    (
        StandardScalerTransform(),
        AddConstTransform(in_column="target", value=10),
        YeoJohnsonTransform(in_column="target"),
        LogTransform(in_column="target", base=2),
        BoxCoxTransform(in_column="target"),
        RobustScalerTransform(in_column="target"),
        MaxAbsScalerTransform(in_column="target"),
        MinMaxScalerTransform(in_column="target"),
    ),
)
def test_dummy_all(toy_dataset_equal_targets_and_quantiles, transform):
    """This test checks that inverse_transform transforms forecast's quantiles the same way with target itself."""
    toy_dataset = toy_dataset_equal_targets_and_quantiles
    _ = transform.fit_transform(toy_dataset.copy())
    toy_dataset = transform.inverse_transform(toy_dataset)

    np.testing.assert_allclose(toy_dataset.iloc[:, 0], toy_dataset.iloc[:, 1])
    np.testing.assert_allclose(toy_dataset.iloc[:, 2], toy_dataset.iloc[:, 3])
