import torch
import torch.nn as nn
import numpy as np

from etna.models.nn.nbeats.blocks import NBeatsBlock
from etna.models.nn.nbeats.blocks import NBeats, GenericBasis, SeasonalityBasis, TrendBasis


class NBeatsInterpretable(nn.Module):
    """Interpretable N-BEATS model."""

    def __init__(self, input_size: int,
                  output_size: int,
                  trend_blocks: int,
                  trend_layers: int,
                  trend_layer_size: int,
                  degree_of_polynomial: int,
                  seasonality_blocks: int,
                  seasonality_layers: int,
                  seasonality_layer_size: int,
                  num_of_harmonics: int):
        """Initialize N-BEATS model.

        Parameters
        ----------
        input_size:
            Input data size.
        output_size:
            Forecast size.
        trend_blocks:
            Number of trend blocks.
        trend_layers:
            Number of inner layers in each trend block.
        trend_layer_size:
            Inner layer size in trend blocks.
        degree_of_polynomial:
            Polynomial degree for trend modeling.
        seasonality_blocks:
            Number of seasonality blocks.
        seasonality_layers:
            Number of inner layers in each seasonality block.
        seasonality_layer_size:
            Inner layer size in seasonality blocks.
        num_of_harmonics:
            Number of harmonics for seasonality estimation.
        """
        super().__init__()

        trend_block = NBeatsBlock(
            input_size=input_size,
            theta_size=2 * (degree_of_polynomial + 1),
            basis_function=TrendBasis(degree=degree_of_polynomial,
                                      backcast_size=input_size,
                                      forecast_size=output_size),
            num_layers=trend_layers,
            layer_size=trend_layer_size)

        seasonality_block = NBeatsBlock(input_size=input_size,
                                        theta_size=4 * int(
                                            np.ceil(num_of_harmonics / 2 * output_size) - (num_of_harmonics - 1)),
                                        basis_function=SeasonalityBasis(harmonics=num_of_harmonics,
                                                                        backcast_size=input_size,
                                                                        forecast_size=output_size),
                                        num_layers=seasonality_layers,
                                        layer_size=seasonality_layer_size)

        # TODO: page 5, 3.3, sharing weights across stacks
        self.model = NBeats(nn.ModuleList(
            [trend_block for _ in range(trend_blocks)] + [seasonality_block for _ in range(seasonality_blocks)]))

    def forward(self, x: torch.Tensor, input_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x:
            Input data.
        input_mask:
            Input mask.

        Returns
        -------
        :
            Forecast data.
        """
        return self.model.forward(x=x, input_mask=input_mask)


class NBeatsGeneric(nn.Module):
    """N-BEATS generic model."""

    def __init__(self, input_size: int, output_size: int,
            stacks: int, layers: int, layer_size: int):
        """Initialize N-BEATS model.

        Parameters
        ----------
        input_size:
            Input data size.
        output_size:
            Forecast size.
        stacks:
            Number of block stacks in model.
        layers:
            Number of inner layers in each block.
        layer_size:
            Inner layers size in blocks.
        """
        super().__init__()

        generic_block = NBeatsBlock(input_size=input_size,
                                               theta_size=input_size + output_size,
                                               basis_function=GenericBasis(backcast_size=input_size,
                                                                           forecast_size=output_size),
                                               num_layers=layers,
                                               layer_size=layer_size)
        # TODO: page 5, 3.3, sharing weights across stacks
        self.model = NBeats(nn.ModuleList([generic_block for _ in range(stacks)]))

    def forward(self, x: torch.Tensor, input_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x:
            Input data.
        input_mask:
            Input mask.

        Returns
        -------
        :
            Forecast data.
        """
        return self.model.forward(x=x, input_mask=input_mask)
