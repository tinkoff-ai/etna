import pytest
import torch
import torch.testing

from etna.models.nn import LevelSSM
from etna.models.nn import LevelTrendSSM
from etna.models.nn import SeasonalitySSM


@pytest.mark.parametrize(
    "ssm, expected_tensor",
    [
        (LevelSSM(), torch.tensor([[[1], [1], [1]], [[1], [1], [1]]]).float()),
        (LevelTrendSSM(), torch.tensor([[[1, 1], [1, 1], [1, 1]], [[1, 1], [1, 1], [1, 1]]]).float()),
        (SeasonalitySSM(num_seasons=2), torch.tensor([[[1, 0], [0, 1], [1, 0]], [[1, 0], [0, 1], [1, 0]]]).float()),
    ],
)
def test_emission_coeff(ssm, expected_tensor, datetime_index=torch.tensor([[0, 1, 0], [0, 1, 0]], dtype=torch.int64)):
    tensor = ssm.emission_coeff(datetime_index)
    assert tensor.shape == expected_tensor.shape
    torch.testing.assert_close(tensor, expected_tensor)


@pytest.mark.parametrize(
    "ssm, expected_tensor",
    [
        (LevelSSM(), torch.tensor([[1]]).float()),
        (LevelTrendSSM(), torch.tensor([[1, 1], [0, 1]]).float()),
        (SeasonalitySSM(num_seasons=2), torch.tensor([[1, 0], [0, 1]]).float()),
    ],
)
def test_transition_coeff(ssm, expected_tensor, datetime_index=torch.tensor([[0, 1, 0], [0, 1, 0]], dtype=torch.int64)):
    tensor = ssm.transition_coeff(datetime_index)
    assert tensor.shape == expected_tensor.shape
    torch.testing.assert_close(tensor, expected_tensor)


@pytest.mark.parametrize(
    "ssm",
    [LevelSSM(), LevelTrendSSM(), SeasonalitySSM(num_seasons=2)],
)
def test_innovation_coeff(ssm, datetime_index=torch.tensor([[0, 1, 0], [0, 1, 0]], dtype=torch.int64)):
    innovation_coeff = ssm.innovation_coeff(datetime_index)
    emission_coeff = ssm.emission_coeff(datetime_index)
    torch.testing.assert_close(innovation_coeff, emission_coeff)
