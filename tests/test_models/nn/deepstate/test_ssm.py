import pytest
import torch
import torch.testing

from etna.models.nn.deepstate import CompositeSSM
from etna.models.nn.deepstate import LevelSSM
from etna.models.nn.deepstate import LevelTrendSSM
from etna.models.nn.deepstate import SeasonalitySSM


@pytest.mark.parametrize(
    "ssm, expected_dim",
    [
        (LevelSSM(), 1),
        (LevelTrendSSM(), 2),
        (SeasonalitySSM(num_seasons=2, timestamp_transform=lambda x: int(x.hour() < 12)), 2),
        (
            CompositeSSM(
                seasonal_ssms=[SeasonalitySSM(num_seasons=2, timestamp_transform=lambda x: int(x.hour() < 12))],
                nonseasonal_ssm=LevelSSM(),
            ),
            3,
        ),
    ],
)
def test_latent_dim(ssm, expected_dim):
    assert ssm.latent_dim() == expected_dim


@pytest.mark.parametrize(
    "ssm, expected_tensor, datetime_index",
    [
        (
            LevelSSM(),
            torch.tensor([[[1], [1], [1]], [[1], [1], [1]]]).float(),
            torch.tensor([[0, 1, 0], [0, 1, 0]], dtype=torch.int64),
        ),
        (
            LevelTrendSSM(),
            torch.tensor([[[1, 1], [1, 1], [1, 1]], [[1, 1], [1, 1], [1, 1]]]).float(),
            torch.tensor([[0, 1, 0], [0, 1, 0]], dtype=torch.int64),
        ),
        (
            SeasonalitySSM(num_seasons=2, timestamp_transform=lambda x: int(x.hour() < 12)),
            torch.tensor([[[1, 0], [0, 1], [1, 0]], [[1, 0], [0, 1], [1, 0]]]).float(),
            torch.tensor([[0, 1, 0], [0, 1, 0]], dtype=torch.int64),
        ),
        (
            CompositeSSM(
                seasonal_ssms=[SeasonalitySSM(num_seasons=2, timestamp_transform=lambda x: int(x.hour() < 12))],
                nonseasonal_ssm=LevelSSM(),
            ),
            torch.tensor([[[1, 0, 1], [0, 1, 1], [1, 0, 1]], [[1, 0, 1], [0, 1, 1], [1, 0, 1]]]).float(),
            torch.tensor([[[0, 1, 0], [0, 1, 0]], [[0, 1, 0], [0, 1, 0]]], dtype=torch.int64),
        ),
    ],
)
def test_emission_coeff(ssm, expected_tensor, datetime_index):
    tensor = ssm.emission_coeff(datetime_index)
    assert tensor.shape == expected_tensor.shape
    torch.testing.assert_close(tensor, expected_tensor)


@pytest.mark.parametrize(
    "ssm, expected_tensor, datetime_index",
    [
        (LevelSSM(), torch.tensor([[1]]).float(), torch.tensor([[0, 1, 0], [0, 1, 0]], dtype=torch.int64)),
        (
            LevelTrendSSM(),
            torch.tensor([[1, 1], [0, 1]]).float(),
            torch.tensor([[0, 1, 0], [0, 1, 0]], dtype=torch.int64),
        ),
        (
            SeasonalitySSM(num_seasons=2, timestamp_transform=lambda x: int(x.hour() < 12)),
            torch.tensor([[1, 0], [0, 1]]).float(),
            torch.tensor([[0, 1, 0], [0, 1, 0]], dtype=torch.int64),
        ),
        (
            CompositeSSM(
                seasonal_ssms=[SeasonalitySSM(num_seasons=2, timestamp_transform=lambda x: int(x.hour() < 12))],
                nonseasonal_ssm=LevelSSM(),
            ),
            torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).float(),
            torch.tensor([[[0, 1, 0], [0, 1, 0]], [[0, 1, 0], [0, 1, 0]]], dtype=torch.int64),
        ),
    ],
)
def test_transition_coeff(ssm, expected_tensor, datetime_index):
    tensor = ssm.transition_coeff(datetime_index)
    assert tensor.shape == expected_tensor.shape
    torch.testing.assert_close(tensor, expected_tensor)


@pytest.mark.parametrize(
    "ssm, datetime_index",
    [
        (LevelSSM(), torch.tensor([[0, 1, 0], [0, 1, 0]], dtype=torch.int64)),
        (LevelTrendSSM(), torch.tensor([[0, 1, 0], [0, 1, 0]], dtype=torch.int64)),
        (
            SeasonalitySSM(num_seasons=2, timestamp_transform=lambda x: int(x.hour() < 12)),
            torch.tensor([[0, 1, 0], [0, 1, 0]], dtype=torch.int64),
        ),
        (
            CompositeSSM(
                seasonal_ssms=[SeasonalitySSM(num_seasons=2, timestamp_transform=lambda x: int(x.hour() < 12))],
                nonseasonal_ssm=LevelSSM(),
            ),
            torch.tensor([[[0, 1, 0], [0, 1, 0]], [[0, 1, 0], [0, 1, 0]]], dtype=torch.int64),
        ),
    ],
)
def test_innovation_coeff(ssm, datetime_index):
    innovation_coeff = ssm.innovation_coeff(datetime_index)
    emission_coeff = ssm.emission_coeff(datetime_index)
    torch.testing.assert_close(innovation_coeff, emission_coeff)
