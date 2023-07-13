import pytest
import torch

from etna.models.nn.nbeats.blocks import GenericBasis
from etna.models.nn.nbeats.blocks import NBeatsBlock
from etna.models.nn.nbeats.blocks import SeasonalityBasis
from etna.models.nn.nbeats.blocks import TrendBasis


@pytest.fixture
def small_tensor():
    return torch.rand(15, 12)


def test_generic_basis_out_format(small_tensor, backcast_size=8, forecast_size=4):
    gb = GenericBasis(backcast_size=backcast_size, forecast_size=forecast_size)
    backcast, forecast = gb(small_tensor)

    assert tuple(backcast.shape) == (small_tensor.shape[0], backcast_size)
    assert tuple(forecast.shape) == (small_tensor.shape[0], forecast_size)


def test_trend_basis_out_format(small_tensor, degree=5, backcast_size=8, forecast_size=4):
    num_terms = degree + 1

    tb = TrendBasis(degree=degree, backcast_size=backcast_size, forecast_size=forecast_size)
    backcast, forecast = tb(small_tensor)

    assert tuple(tb.backcast_time.shape) == (num_terms, backcast_size)
    assert tuple(tb.forecast_time.shape) == (num_terms, forecast_size)

    assert tuple(backcast.shape) == (small_tensor.shape[0], backcast_size)
    assert tuple(forecast.shape) == (small_tensor.shape[0], forecast_size)


def test_seasonality_basis_out_format(small_tensor, harmonics=1, backcast_size=6, forecast_size=6):
    params_per_harmonic = small_tensor.shape[1] // 4

    sb = SeasonalityBasis(harmonics=harmonics, backcast_size=backcast_size, forecast_size=forecast_size)
    backcast, forecast = sb(small_tensor)

    assert tuple(sb.backcast_cos_template.shape) == (params_per_harmonic, backcast_size)
    assert tuple(sb.backcast_sin_template.shape) == (params_per_harmonic, backcast_size)

    assert tuple(sb.forecast_cos_template.shape) == (params_per_harmonic, forecast_size)
    assert tuple(sb.forecast_sin_template.shape) == (params_per_harmonic, forecast_size)

    assert tuple(backcast.shape) == (small_tensor.shape[0], backcast_size)
    assert tuple(forecast.shape) == (small_tensor.shape[0], forecast_size)


@pytest.mark.parametrize(
    "theta_size,basis_function",
    (
        (10, GenericBasis(backcast_size=6, forecast_size=6)),
        (6, TrendBasis(degree=2, backcast_size=6, forecast_size=6)),
        (12, SeasonalityBasis(harmonics=1, backcast_size=6, forecast_size=6)),
    ),
)
def test_nbeats_block_out_format(small_tensor, theta_size, basis_function, num_layers=2, layer_size=32):
    block = NBeatsBlock(
        input_size=small_tensor.shape[1],
        theta_size=theta_size,
        basis_function=basis_function,
        num_layers=num_layers,
        layer_size=layer_size,
    )

    backcast, forecast = block(small_tensor)

    assert tuple(backcast.shape) == (small_tensor.shape[0], 6)
    assert tuple(forecast.shape) == (small_tensor.shape[0], 6)
