from abc import abstractmethod
from functools import partial
from typing import Dict
from typing import Literal
from typing import Optional
from typing import Union

import numpy as np

from etna import SETTINGS
from etna.distributions import BaseDistribution
from etna.distributions import FloatDistribution
from etna.distributions import IntDistribution

if SETTINGS.torch_required:
    import torch

    from etna.models.base import DeepBaseModel
    from etna.models.nn.nbeats.metrics import NBeatsLoss
    from etna.models.nn.nbeats.nets import NBeatsBaseNet
    from etna.models.nn.nbeats.nets import NBeatsGenericNet
    from etna.models.nn.nbeats.nets import NBeatsInterpretableNet
    from etna.models.nn.nbeats.utils import _create_or_update
    from etna.models.nn.nbeats.utils import prepare_test_batch
    from etna.models.nn.nbeats.utils import prepare_train_batch


class NBeatsBaseModel(DeepBaseModel):
    """Base class for N-BEATS models."""

    @abstractmethod
    def __init__(
        self,
        net: "NBeatsBaseNet",
        window_sampling_limit: Optional[int] = None,
        train_batch_size: int = 1024,
        test_batch_size: int = 1024,
        trainer_params: Optional[dict] = None,
        train_dataloader_params: Optional[dict] = None,
        test_dataloader_params: Optional[dict] = None,
        val_dataloader_params: Optional[dict] = None,
        split_params: Optional[dict] = None,
        random_state: Optional[int] = None,
    ):
        gen_state = np.random.RandomState(seed=random_state)
        train_collate_fn = partial(
            prepare_train_batch,
            input_size=net.input_size,
            output_size=net.output_size,
            window_sampling_limit=window_sampling_limit,
            random_state=gen_state,
        )
        val_collate_fn = partial(
            prepare_train_batch,
            input_size=net.input_size,
            output_size=net.output_size,
            window_sampling_limit=window_sampling_limit,
            random_state=gen_state,
        )
        test_collate_fn = partial(prepare_test_batch, input_size=net.input_size)

        train_dataloader_params = _create_or_update(
            param=train_dataloader_params, name="collate_fn", value=train_collate_fn
        )
        val_dataloader_params = _create_or_update(param=val_dataloader_params, name="collate_fn", value=val_collate_fn)
        test_dataloader_params = _create_or_update(
            param=test_dataloader_params, name="collate_fn", value=test_collate_fn
        )

        if trainer_params is None or "gradient_clip_val" not in trainer_params:
            trainer_params = _create_or_update(param=trainer_params, name="gradient_clip_val", value=1.0)

        super().__init__(
            net=net,
            encoder_length=net.input_size,
            decoder_length=net.output_size,
            train_batch_size=train_batch_size,
            test_batch_size=test_batch_size,
            train_dataloader_params=train_dataloader_params,
            test_dataloader_params=test_dataloader_params,
            val_dataloader_params=val_dataloader_params,
            trainer_params=trainer_params,
            split_params=split_params,
        )


class NBeatsInterpretableModel(NBeatsBaseModel):
    """Interpretable N-BEATS model.

    Paper: https://arxiv.org/pdf/1905.10437.pdf

    Official implementation: https://github.com/ServiceNow/N-BEATS
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        loss: Union[Literal["mse"], Literal["mae"], Literal["smape"], Literal["mape"], "torch.nn.Module"] = "mse",
        trend_blocks: int = 3,
        trend_layers: int = 4,
        trend_layer_size: int = 256,
        degree_of_polynomial: int = 2,
        seasonality_blocks: int = 3,
        seasonality_layers: int = 4,
        seasonality_layer_size: int = 2048,
        num_of_harmonics: int = 1,
        lr: float = 0.001,
        window_sampling_limit: Optional[int] = None,
        optimizer_params: Optional[dict] = None,
        train_batch_size: int = 1024,
        test_batch_size: int = 1024,
        trainer_params: Optional[dict] = None,
        train_dataloader_params: Optional[dict] = None,
        test_dataloader_params: Optional[dict] = None,
        val_dataloader_params: Optional[dict] = None,
        split_params: Optional[dict] = None,
        random_state: Optional[int] = None,
    ):
        """Init interpretable N-BEATS model.

        Parameters
        ----------
        input_size:
            Input data size.
        output_size:
            Forecast size.
        loss:
            Optimisation objective. The loss function should accept three arguments: ``y_true``, ``y_pred`` and ``mask``.
            The last parameter is a binary mask that denotes which points are valid forecasts.
            There are several implemented loss functions available in the :mod:`etna.models.nn.nbeats.metrics` module.
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
        window_sampling_limit:
            Size of history for sampling training data. If set to ``None`` full series history used for sampling.
        optimizer_params:
            Additional parameters for the optimizer.
        train_batch_size:
            Batch size for training.
        test_batch_size:
            Batch size for testing.
        optimizer_params:
            Parameters for optimizer for Adam optimizer (api reference :py:class:`torch.optim.Adam`).
        trainer_params:
            Pytorch lightning  trainer parameters (api reference :py:class:`pytorch_lightning.trainer.trainer.Trainer`).
        train_dataloader_params:
            Parameters for train dataloader like sampler for example (api reference :py:class:`torch.utils.data.DataLoader`).
        test_dataloader_params:
            Parameters for test dataloader.
        val_dataloader_params:
            Parameters for validation dataloader.
        split_params:
            Dictionary with parameters for :py:func:`torch.utils.data.random_split` for train-test splitting
                * **train_size**: (*float*) value from 0 to 1 - fraction of samples to use for training

                * **generator**: (*Optional[torch.Generator]*) - generator for reproducibile train-test splitting

                * **torch_dataset_size**: (*Optional[int]*) - number of samples in dataset, in case of dataset not implementing ``__len__``
        random_state:
            Random state for train batches generation.
        """
        if isinstance(loss, str):
            try:
                self.loss = NBeatsLoss[loss].value

            except KeyError as e:
                raise NotImplementedError(
                    f"{e} is not a valid {NBeatsLoss.__name__}. "
                    f"Only {', '.join([repr(m.name) for m in NBeatsLoss])} loss name allowed"
                )

        else:
            self.loss = loss

        self.input_size = input_size
        self.output_size = output_size
        self.trend_blocks = trend_blocks
        self.trend_layers = trend_layers
        self.trend_layer_size = trend_layer_size
        self.degree_of_polynomial = degree_of_polynomial
        self.seasonality_blocks = seasonality_blocks
        self.seasonality_layers = seasonality_layers
        self.seasonality_layer_size = seasonality_layer_size
        self.num_of_harmonics = num_of_harmonics
        self.lr = lr
        self.window_sampling_limit = window_sampling_limit
        self.optimizer_params = optimizer_params
        self.random_state = random_state

        super().__init__(
            net=NBeatsInterpretableNet(
                input_size=input_size,
                output_size=output_size,
                trend_blocks=trend_blocks,
                trend_layers=trend_layers,
                trend_layer_size=trend_layer_size,
                degree_of_polynomial=degree_of_polynomial,
                seasonality_blocks=seasonality_blocks,
                seasonality_layers=seasonality_layers,
                seasonality_layer_size=seasonality_layer_size,
                num_of_harmonics=num_of_harmonics,
                lr=lr,
                loss=self.loss,
                optimizer_params=optimizer_params,
            ),
            window_sampling_limit=window_sampling_limit,
            train_batch_size=train_batch_size,
            test_batch_size=test_batch_size,
            train_dataloader_params=train_dataloader_params,
            test_dataloader_params=test_dataloader_params,
            val_dataloader_params=val_dataloader_params,
            trainer_params=trainer_params,
            split_params=split_params,
            random_state=random_state,
        )

    def params_to_tune(self) -> Dict[str, BaseDistribution]:
        """Get default grid for tuning hyperparameters.

        This grid tunes parameters: ``trend_blocks``, ``trend_layers``, ``trend_layer_size``, ``degree_of_polynomial``,
        ``seasonality_blocks``, ``seasonality_layers``, ``seasonality_layer_size``, ``lr``.
        Other parameters are expected to be set by the user.

        Returns
        -------
        :
            Grid to tune.
        """
        return {
            "trend_blocks": IntDistribution(low=1, high=10),
            "trend_layers": IntDistribution(low=1, high=10),
            "trend_layer_size": IntDistribution(low=4, high=1024, step=4),
            "degree_of_polynomial": IntDistribution(low=0, high=4),
            "seasonality_blocks": IntDistribution(low=1, high=10),
            "seasonality_layers": IntDistribution(low=1, high=10),
            "seasonality_layer_size": IntDistribution(low=8, high=4096, step=8),
            "lr": FloatDistribution(low=1e-5, high=1e-2, log=True),
        }


class NBeatsGenericModel(NBeatsBaseModel):
    """Generic N-BEATS model.

    Paper: https://arxiv.org/pdf/1905.10437.pdf

    Official implementation: https://github.com/ServiceNow/N-BEATS
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        loss: Union[Literal["mse"], Literal["mae"], Literal["smape"], Literal["mape"], "torch.nn.Module"] = "mse",
        stacks: int = 30,
        layers: int = 4,
        layer_size: int = 512,
        lr: float = 0.001,
        window_sampling_limit: Optional[int] = None,
        optimizer_params: Optional[dict] = None,
        train_batch_size: int = 1024,
        test_batch_size: int = 1024,
        trainer_params: Optional[dict] = None,
        train_dataloader_params: Optional[dict] = None,
        test_dataloader_params: Optional[dict] = None,
        val_dataloader_params: Optional[dict] = None,
        split_params: Optional[dict] = None,
        random_state: Optional[int] = None,
    ):
        """Init generic N-BEATS model.

        Parameters
        ----------
        input_size:
            Input data size.
        output_size:
            Forecast size.
        loss:
            Optimisation objective. The loss function should accept three arguments: ``y_true``, ``y_pred`` and ``mask``.
            The last parameter is a binary mask that denotes which points are valid forecasts.
            There are several implemented loss functions available in the :mod:`etna.models.nn.nbeats.metrics` module.
        stacks:
            Number of block stacks in model.
        layers:
            Number of inner layers in each block.
        layer_size:
            Inner layers size in blocks.
        lr:
            Optimizer learning rate.
        window_sampling_limit:
            Size of history for sampling training data. If set to ``None`` full series history used for sampling.
        optimizer_params:
            Additional parameters for the optimizer.
        train_batch_size:
            Batch size for training.
        test_batch_size:
            Batch size for testing.
        optimizer_params:
            Parameters for optimizer for Adam optimizer (api reference :py:class:`torch.optim.Adam`).
        trainer_params:
            Pytorch ligthning  trainer parameters (api reference :py:class:`pytorch_lightning.trainer.trainer.Trainer`).
        train_dataloader_params:
            Parameters for train dataloader like sampler for example (api reference :py:class:`torch.utils.data.DataLoader`).
        test_dataloader_params:
            Parameters for test dataloader.
        val_dataloader_params:
            Parameters for validation dataloader.
        split_params:
            Dictionary with parameters for :py:func:`torch.utils.data.random_split` for train-test splitting
                * **train_size**: (*float*) value from 0 to 1 - fraction of samples to use for training

                * **generator**: (*Optional[torch.Generator]*) - generator for reproducibile train-test splitting

                * **torch_dataset_size**: (*Optional[int]*) - number of samples in dataset, in case of dataset not implementing ``__len__``
        random_state:
            Random state for train batches generation.
        """
        if isinstance(loss, str):
            try:
                self.loss = NBeatsLoss[loss].value

            except KeyError as e:
                raise NotImplementedError(
                    f"{e} is not a valid {NBeatsLoss.__name__}. "
                    f"Only {', '.join([repr(m.name) for m in NBeatsLoss])} loss name allowed"
                )

        else:
            self.loss = loss

        self.input_size = input_size
        self.output_size = output_size
        self.stacks = stacks
        self.layers = layers
        self.layer_size = layer_size
        self.lr = lr
        self.window_sampling_limit = window_sampling_limit
        self.optimizer_params = optimizer_params
        self.random_state = random_state

        super().__init__(
            net=NBeatsGenericNet(
                input_size=input_size,
                output_size=output_size,
                stacks=stacks,
                layers=layers,
                layer_size=layer_size,
                lr=lr,
                loss=self.loss,
                optimizer_params=optimizer_params,
            ),
            window_sampling_limit=window_sampling_limit,
            train_batch_size=train_batch_size,
            test_batch_size=test_batch_size,
            train_dataloader_params=train_dataloader_params,
            test_dataloader_params=test_dataloader_params,
            val_dataloader_params=val_dataloader_params,
            trainer_params=trainer_params,
            split_params=split_params,
            random_state=random_state,
        )

    def params_to_tune(self) -> Dict[str, BaseDistribution]:
        """Get default grid for tuning hyperparameters.

        This grid tunes parameters: ``stacks``, ``layers``, ``lr``, ``layer_size``.
        Other parameters are expected to be set by the user.

        Returns
        -------
        :
            Grid to tune.
        """
        return {
            "stacks": IntDistribution(low=1, high=40),
            "layers": IntDistribution(low=1, high=8),
            "layer_size": IntDistribution(low=4, high=1024, step=4),
            "lr": FloatDistribution(low=1e-5, high=1e-2, log=True),
        }
