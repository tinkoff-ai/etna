from typing import Tuple

import numpy as np

from etna import SETTINGS

if SETTINGS.torch_required:
    import torch
    import torch.nn as nn


class NBeatsBlock(nn.Module):
    """Base N-BEATS block which takes a basis function as an argument."""

    def __init__(self, input_size: int, theta_size: int, basis_function: "nn.Module", num_layers: int, layer_size: int):
        """N-BEATS block.

        Parameters
        ----------
        input_size:
            In-sample size.
        theta_size:
            Number of parameters for the basis function.
        basis_function:
            Basis function which takes the parameters and produces backcast and forecast.
        num_layers:
            Number of layers.
        layer_size
            Layer size.
        """
        super().__init__()

        layers = [nn.Linear(in_features=input_size, out_features=layer_size), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_features=layer_size, out_features=layer_size))
            layers.append(nn.ReLU())

        self.layers = nn.ModuleList(layers)

        self.basis_parameters = nn.Linear(in_features=layer_size, out_features=theta_size)
        self.basis_function = basis_function

    def forward(self, x: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
        """Forward pass.

        Parameters
        ----------
        x:
            Input data.

        Returns
        -------
        :
            Tuple with backcast and forecast.
        """
        for layer in self.layers:
            x = layer(x)

        basis_parameters = self.basis_parameters(x)
        return self.basis_function(basis_parameters)


class GenericBasis(nn.Module):
    """Generic basis function."""

    def __init__(self, backcast_size: int, forecast_size: int):
        """Initialize generic basis function.

        Parameters
        ----------
        backcast_size:
            Number of backcast values.
        forecast_size:
            Number of forecast values.
        """
        super().__init__()
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size

    def forward(self, theta: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
        """Forward pass.

        Parameters
        ----------
        theta:
            Basis function parameters.

        Returns
        -------
        :
            Tuple with backcast and forecast.
        """
        return theta[:, : self.backcast_size], theta[:, -self.forecast_size :]


class TrendBasis(nn.Module):
    """Polynomial trend basis function."""

    def __init__(self, degree: int, backcast_size: int, forecast_size: int):
        """Initialize trend basis function.

        Parameters
        ----------
        degree:
            Degree of polynomial for trend modeling.
        backcast_size:
            Number of backcast values.
        forecast_size:
            Number of forecast values.
        """
        super().__init__()
        self.num_poly_terms = degree + 1

        self.backcast_time = nn.Parameter(self._trend_tensor(size=backcast_size), requires_grad=False)
        self.forecast_time = nn.Parameter(self._trend_tensor(size=forecast_size), requires_grad=False)

    def _trend_tensor(self, size: int) -> "torch.Tensor":
        """Prepare trend tensor."""
        time = torch.arange(size) / size
        degrees = torch.arange(self.num_poly_terms)
        trend_tensor = torch.transpose(time[:, None] ** degrees[None], 0, 1)
        return trend_tensor

    def forward(self, theta: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
        """Forward pass.

        Parameters
        ----------
        theta:
            Basis function parameters.

        Returns
        -------
        :
            Tuple with backcast and forecast.
        """
        backcast = theta[:, : self.num_poly_terms] @ self.backcast_time
        forecast = theta[:, self.num_poly_terms :] @ self.forecast_time
        return backcast, forecast


class SeasonalityBasis(nn.Module):
    """Harmonic seasonality basis function."""

    def __init__(self, harmonics: int, backcast_size: int, forecast_size: int):
        """Initialize seasonality basis function.

        Parameters
        ----------
        harmonics:
            Harmonics range.
        backcast_size:
            Number of backcast values.
        forecast_size:
            Number of forecast values.
        """
        super().__init__()

        freq = torch.arange(harmonics - 1, harmonics / 2 * forecast_size) / harmonics
        freq[0] = 0.0
        frequency = torch.unsqueeze(freq, 0)

        backcast_grid = -2 * np.pi * torch.arange(backcast_size)[:, None] / forecast_size
        backcast_grid = backcast_grid * frequency

        forecast_grid = 2 * np.pi * torch.arange(forecast_size)[:, None] / forecast_size
        forecast_grid = forecast_grid * frequency

        self.backcast_cos_template = nn.Parameter(torch.transpose(torch.cos(backcast_grid), 0, 1), requires_grad=False)
        self.backcast_sin_template = nn.Parameter(torch.transpose(torch.sin(backcast_grid), 0, 1), requires_grad=False)
        self.forecast_cos_template = nn.Parameter(torch.transpose(torch.cos(forecast_grid), 0, 1), requires_grad=False)
        self.forecast_sin_template = nn.Parameter(torch.transpose(torch.sin(forecast_grid), 0, 1), requires_grad=False)

    def forward(self, theta: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
        """Forward pass.

        Parameters
        ----------
        theta:
            Basis function parameters.

        Returns
        -------
        :
            Tuple with backcast and forecast.
        """
        params_per_harmonic = theta.shape[1] // 4

        backcast_harmonics_cos = theta[:, :params_per_harmonic] @ self.backcast_cos_template
        backcast_harmonics_sin = theta[:, params_per_harmonic : 2 * params_per_harmonic] @ self.backcast_sin_template
        backcast = backcast_harmonics_sin + backcast_harmonics_cos

        forecast_harmonics_cos = (
            theta[:, 2 * params_per_harmonic : 3 * params_per_harmonic] @ self.forecast_cos_template
        )
        forecast_harmonics_sin = theta[:, 3 * params_per_harmonic :] @ self.forecast_sin_template
        forecast = forecast_harmonics_sin + forecast_harmonics_cos

        return backcast, forecast


class NBeats(nn.Module):
    """N-BEATS model."""

    def __init__(self, blocks: "nn.ModuleList"):
        """Initialize N-BEATS model.

        Parameters
        ----------
        blocks:
            Model blocks.
        """
        super().__init__()
        self.blocks = blocks

    def forward(self, x: "torch.Tensor", input_mask: "torch.Tensor") -> "torch.Tensor":
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
            Forecast tensor.
        """
        residuals = x.flip(dims=(1,))
        input_mask = input_mask.flip(dims=(1,))
        forecast = x[:, -1:]

        for block in self.blocks:
            backcast, block_forecast = block(residuals)
            residuals = (residuals - backcast) * input_mask
            forecast = forecast + block_forecast

        return forecast
