from abc import abstractmethod
from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import pandas as pd
from typing_extensions import TypedDict

from etna import SETTINGS

if SETTINGS.torch_required:
    import torch
    import torch.nn as nn

    from etna.models.base import DeepBaseNet
    from etna.models.nn.nbeats.blocks import GenericBasis
    from etna.models.nn.nbeats.blocks import NBeats
    from etna.models.nn.nbeats.blocks import NBeatsBlock
    from etna.models.nn.nbeats.blocks import SeasonalityBasis
    from etna.models.nn.nbeats.blocks import TrendBasis


class NBeatsBatch(TypedDict):
    """Batch specification for N-BEATS."""

    history: "torch.Tensor"
    history_mask: "torch.Tensor"
    target: "torch.Tensor"
    target_mask: "torch.Tensor"
    segment: "torch.Tensor"


class NBeatsBaseNet(DeepBaseNet):
    """Base class for N-BEATS models."""

    @abstractmethod
    def __init__(
        self,
        model: "nn.Module",
        input_size: int,
        output_size: int,
        loss: "nn.Module",
        lr: float,
        optimizer_params: Optional[Dict[str, Any]],
    ):
        super().__init__()

        self.model = model
        self.input_size = input_size
        self.output_size = output_size
        self.loss = loss
        self.lr = lr
        self.optimizer_params = optimizer_params if optimizer_params is not None else {}

    def forward(self, batch: NBeatsBatch) -> "torch.Tensor":
        """Forward pass.

        Parameters
        ----------
        batch:
            Batch of input data.

        Returns
        -------
        :
            Prediction data.
        """
        history = batch["history"].float()
        history_mask = batch["history_mask"].float()
        return self.model(x=history, input_mask=history_mask).reshape(-1, self.output_size, 1)

    def step(self, batch: NBeatsBatch, *args, **kwargs):  # type: ignore
        """Step for loss computation for training or validation.

        Parameters
        ----------
        batch:
            Batch of input data.

        Returns
        -------
        :
            loss, true_target, prediction_target
        """
        history = batch["history"].float()
        history_mask = batch["history_mask"].float()
        target = batch["target"].float()
        target_mask = batch["target_mask"].float()

        forecast = self.model(x=history, input_mask=history_mask)
        loss = self.loss(target, forecast, target_mask)

        return loss, target, forecast

    def make_samples(self, df: pd.DataFrame, encoder_length: int, decoder_length: int) -> Iterable[dict]:
        """Make samples from segment DataFrame."""
        values_target = df["target"].values
        segment = df["segment"].values[0]

        sample: Dict[str, Any] = {
            "history": values_target,
            "history_mask": None,
            "target": None,
            "target_mask": None,
            "segment": segment,
        }
        yield sample

    def configure_optimizers(self) -> Tuple[List["torch.optim.Optimizer"], List[Dict[str, Any]]]:
        """Optimizer configuration."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, **self.optimizer_params)
        epochs = self.optimizer_params.get("max_epochs", 1000)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda i: 0.5 ** (3 * i // epochs))
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]


class NBeatsInterpretableNet(NBeatsBaseNet):
    """Interpretable N-BEATS model."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        loss: "torch.nn.Module",
        trend_blocks: int,
        trend_layers: int,
        trend_layer_size: int,
        degree_of_polynomial: int,
        seasonality_blocks: int,
        seasonality_layers: int,
        seasonality_layer_size: int,
        num_of_harmonics: int,
        lr: float,
        optimizer_params: Optional[Dict[str, Any]] = None,
    ):
        """Initialize N-BEATS model.

        Parameters
        ----------
        input_size:
            Input data size.
        output_size:
            Forecast size.
        loss:
            Optimisation objective. The loss function should accept three arguments: ``y_true``, ``y_pred`` and ``mask``.
            The last parameter is a binary mask that denotes which points are valid forecasts.
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
        lr:
            Optimizer learning rate.
        optimizer_params:
            Additional parameters for the optimizer.
        """
        self.trend_blocks = trend_blocks
        self.trend_layers = trend_layers
        self.trend_layer_size = trend_layer_size
        self.degree_of_polynomial = degree_of_polynomial
        self.seasonality_blocks = seasonality_blocks
        self.seasonality_layers = seasonality_layers
        self.seasonality_layer_size = seasonality_layer_size
        self.num_of_harmonics = num_of_harmonics

        trend_block = NBeatsBlock(
            input_size=input_size,
            theta_size=2 * (degree_of_polynomial + 1),
            basis_function=TrendBasis(degree=degree_of_polynomial, backcast_size=input_size, forecast_size=output_size),
            num_layers=trend_layers,
            layer_size=trend_layer_size,
        )

        seasonality_block = NBeatsBlock(
            input_size=input_size,
            theta_size=4 * int(np.ceil(num_of_harmonics / 2 * output_size) - (num_of_harmonics - 1)),
            basis_function=SeasonalityBasis(
                harmonics=num_of_harmonics, backcast_size=input_size, forecast_size=output_size
            ),
            num_layers=seasonality_layers,
            layer_size=seasonality_layer_size,
        )

        model = NBeats(
            nn.ModuleList(
                [trend_block for _ in range(trend_blocks)] + [seasonality_block for _ in range(seasonality_blocks)]
            )
        )

        super().__init__(
            model=model,
            input_size=input_size,
            output_size=output_size,
            loss=loss,
            lr=lr,
            optimizer_params=optimizer_params,
        )


class NBeatsGenericNet(NBeatsBaseNet):
    """N-BEATS generic model."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        loss: "nn.Module",
        stacks: int,
        layers: int,
        layer_size: int,
        lr: float,
        optimizer_params: Optional[Dict[str, Any]] = None,
    ):
        """Initialize N-BEATS model.

        Parameters
        ----------
        input_size:
            Input data size.
        output_size:
            Forecast size.
        loss:
            Optimisation objective. The loss function should accept three arguments: ``y_true``, ``y_pred`` and ``mask``.
            The last parameter is a binary mask that denotes which points are valid forecasts.
        stacks:
            Number of block stacks in model.
        layers:
            Number of inner layers in each block.
        layer_size:
            Inner layers size in blocks.
        lr:
            Optimizer learning rate.
        optimizer_params:
            Additional parameters for the optimizer.
        """
        self.stacks = stacks
        self.layers = layers
        self.layer_size = layer_size

        generic_block = NBeatsBlock(
            input_size=input_size,
            theta_size=input_size + output_size,
            basis_function=GenericBasis(backcast_size=input_size, forecast_size=output_size),
            num_layers=layers,
            layer_size=layer_size,
        )

        model = NBeats(nn.ModuleList([generic_block for _ in range(stacks)]))

        super().__init__(
            model=model,
            input_size=input_size,
            output_size=output_size,
            loss=loss,
            lr=lr,
            optimizer_params=optimizer_params,
        )
