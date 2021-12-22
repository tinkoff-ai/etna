import numpy as np
import pandas as pd
import pytest

from etna.datasets import TSDataset
from etna.transforms import AddConstTransform
from etna.transforms import BinsegTrendTransform
from etna.transforms import LinearTrendTransform
from etna.transforms import LogTransform
from etna.transforms import StandardScalerTransform
from etna.transforms import STLTransform
from etna.transforms import YeoJohnsonTransform


@pytest.fixture
def toy_dataset_equal_targets_and_quantiles():
    n_periods = 5
    n_segments = 2

    time = list(pd.date_range("2020-01-01", periods=n_periods, freq="1D"))

    df = {
        "timestamp": time * n_segments,
        "segment": ["a"] * n_periods + ["b"] * n_periods,
        "target": np.concatenate((np.array((2, 3, 4, 5, 5)), np.array((3, 3, 3, 5, 2)))).astype(np.float64),
        "target_0.01": np.concatenate((np.array((2, 3, 4, 5, 5)), np.array((3, 3, 3, 5, 2)))).astype(np.float64),
    }
    return TSDataset.to_dataset(pd.DataFrame(df))


@pytest.fixture
def toy_dataset_with_mean_shift_in_target():
    mean_1 = 10
    mean_2 = 20
    n_periods = 5
    n_segments = 2

    time = list(pd.date_range("2020-01-01", periods=n_periods, freq="1D"))

    df = {
        "timestamp": time * n_segments,
        "segment": ["a"] * n_periods + ["b"] * n_periods,
        "target": np.concatenate((np.array((-1, 3, 3, -4, -1)) + mean_1, np.array((-2, 3, -4, 5, -2)) + mean_2)).astype(
            np.float64
        ),
        "target_0.01": np.concatenate((np.array((-1, 3, 3, -4, -1)), np.array((-2, 3, -4, 5, -2)))).astype(np.float64),
    }
    return TSDataset.to_dataset(pd.DataFrame(df))


def test_standard_scaler_dummy_mean_shift_for_quantiles_per_segment(toy_dataset_with_mean_shift_in_target):
    toy_dataset = toy_dataset_with_mean_shift_in_target
    scaler = StandardScalerTransform(in_column="target", with_std=False)
    toy_dataset = scaler.fit_transform(toy_dataset)
    toy_dataset = scaler.inverse_transform(toy_dataset)
    np.testing.assert_allclose(toy_dataset.iloc[:, 0], toy_dataset.iloc[:, 1])
    np.testing.assert_allclose(toy_dataset.iloc[:, 2], toy_dataset.iloc[:, 3])


def test_standard_scaler_dummy_mean_shift_for_quantiles_macro(toy_dataset_with_mean_shift_in_target):
    toy_dataset = toy_dataset_with_mean_shift_in_target

    scaler = StandardScalerTransform(in_column="target", with_std=False, mode="macro")
    mean_1 = toy_dataset.iloc[:, 0].mean()
    mean_2 = toy_dataset.iloc[:, 2].mean()
    toy_dataset = scaler.fit_transform(toy_dataset)
    toy_dataset = scaler.inverse_transform(toy_dataset)
    np.testing.assert_allclose(toy_dataset.iloc[:, 0], toy_dataset.iloc[:, 1] - (mean_1 + mean_2) / 2 + mean_1)
    np.testing.assert_allclose(toy_dataset.iloc[:, 2], toy_dataset.iloc[:, 3] - (mean_1 + mean_2) / 2 + mean_2)


def test_add_constant_dummy(toy_dataset_equal_targets_and_quantiles):
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
        BinsegTrendTransform(in_column="target", n_bkps=1, min_size=1, model="l2"),
        STLTransform(in_column="target", period=2),
        YeoJohnsonTransform(in_column="target"),
        LinearTrendTransform(in_column="target"),
        LogTransform(in_column="target", base=2),
    ),
)
def test_dummy_all(toy_dataset_equal_targets_and_quantiles, transform):
    toy_dataset = toy_dataset_equal_targets_and_quantiles
    _ = transform.fit_transform(toy_dataset.copy())
    toy_dataset = transform.inverse_transform(toy_dataset)

    np.testing.assert_allclose(toy_dataset.iloc[:, 0], toy_dataset.iloc[:, 1])
    np.testing.assert_allclose(toy_dataset.iloc[:, 2], toy_dataset.iloc[:, 3])
