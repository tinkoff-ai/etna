import pytest
from torch import Tensor

from etna.models.nn import LevelSSM
from etna.models.nn import LevelTrendSSM
from etna.models.nn import SeasonalitySSM


@pytest.mark.parametrize(
    "ssm, expected_tensor",
    [(LevelSSM(), Tensor([[[1],[1],[1]],[[1],[1],[1]]]),
     (LevelTrendSSM(), Tensor()),
     (SeasonalitySSM(num_seasons=2), Tensor())
     ],
)
def test_emission_coeff(ssm, expected_tensor, datetime_index = Tensor([[1,2,1],[1,2,1]])):
    tensor = ssm.emission_coeff(datetime_index)
    assert tensor.shape == expected_tensor.shape
    assert tensor == expected_tensor
