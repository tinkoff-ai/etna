from abc import abstractmethod
from typing import Optional

from etna import SETTINGS

if SETTINGS.torch_required:
    import torch

    from etna.models.base import DeepBaseModel
    from etna.models.nn.nbeats.models import NBeatsBaseNet
    from etna.models.nn.nbeats.models import NBeatsGenericNet
    from etna.models.nn.nbeats.models import NBeatsInterpretableNet
    from etna.models.nn.nbeats.utils import _create_or_update
    from etna.models.nn.nbeats.utils import prepare_test_batch
    from etna.models.nn.nbeats.utils import prepare_train_batch


class NBeatsBaseModel(DeepBaseModel):
    """Base class for N-BEATS models."""

    @abstractmethod
    def __init__(
        self,
        net: "NBeatsBaseNet",
        train_batch_size: int = 1024,
        test_batch_size: int = 1024,
        trainer_params: Optional[dict] = None,
        train_dataloader_params: Optional[dict] = None,
        test_dataloader_params: Optional[dict] = None,
        val_dataloader_params: Optional[dict] = None,
        split_params: Optional[dict] = None,
    ):
        def train_collate_fn(data):
            return prepare_train_batch(
                data=data, batch_size=train_batch_size, input_size=net.input_size, output_size=net.output_size
            )

        def val_collate_fn(data):
            return prepare_train_batch(
                data=data, batch_size=test_batch_size, input_size=net.input_size, output_size=net.output_size
            )

        def test_collate_fn(data):
            return prepare_test_batch(data=data, batch_size=test_batch_size, input_size=net.input_size)

        train_dataloader_params = _create_or_update(
            param=train_dataloader_params, name="collate_fn", value=train_collate_fn
        )
        val_dataloader_params = _create_or_update(param=val_dataloader_params, name="collate_fn", value=val_collate_fn)
        test_dataloader_params = _create_or_update(
            param=test_dataloader_params, name="collate_fn", value=test_collate_fn
        )

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
    """Interpretable N-BEATS model."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        loss: "torch.nn.Module",
        trend_blocks: int = 3,
        trend_layers: int = 4,
        trend_layer_size: int = 256,
        degree_of_polynomial: int = 2,
        seasonality_blocks: int = 3,
        seasonality_layers: int = 4,
        seasonality_layer_size: int = 2048,
        num_of_harmonics: int = 1,
        lr: float = 0.001,
        optimizer_params: Optional[dict] = None,
        train_batch_size: int = 1024,
        test_batch_size: int = 1024,
        trainer_params: Optional[dict] = None,
        train_dataloader_params: Optional[dict] = None,
        test_dataloader_params: Optional[dict] = None,
        val_dataloader_params: Optional[dict] = None,
        split_params: Optional[dict] = None,
    ):
        """Init interpretable N-BEATS model.

        Parameters
        ----------
        input_size:
            Input data size.
        output_size:
            Forecast size.
        loss:
            Optimization objective.
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
        train_batch_size:
            batch size for training
        test_batch_size:
            batch size for testing
        optimizer_params:
            parameters for optimizer for Adam optimizer (api reference :py:class:`torch.optim.Adam`)
        trainer_params:
            Pytorch lightning  trainer parameters (api reference :py:class:`pytorch_lightning.trainer.trainer.Trainer`)
        train_dataloader_params:
            parameters for train dataloader like sampler for example (api reference :py:class:`torch.utils.data.DataLoader`)
        test_dataloader_params:
            parameters for test dataloader
        val_dataloader_params:
            parameters for validation dataloader
        split_params:
            dictionary with parameters for :py:func:`torch.utils.data.random_split` for train-test splitting
                * **train_size**: (*float*) value from 0 to 1 - fraction of samples to use for training

                * **generator**: (*Optional[torch.Generator]*) - generator for reproducibile train-test splitting

                * **torch_dataset_size**: (*Optional[int]*) - number of samples in dataset, in case of dataset not implementing ``__len__``
        """
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
        self.loss = loss
        self.optimizer_params = optimizer_params

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
                loss=loss,
                optimizer_params=optimizer_params,
            ),
            train_batch_size=train_batch_size,
            test_batch_size=test_batch_size,
            train_dataloader_params=train_dataloader_params,
            test_dataloader_params=test_dataloader_params,
            val_dataloader_params=val_dataloader_params,
            trainer_params=trainer_params,
            split_params=split_params,
        )


class NBeatsGenericModel(NBeatsBaseModel):
    """Generic N-BEATS model."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        loss: "torch.nn.Module",
        stacks: int = 30,
        layers: int = 4,
        layer_size: int = 512,
        lr: float = 0.001,
        optimizer_params: Optional[dict] = None,
        train_batch_size: int = 1024,
        test_batch_size: int = 1024,
        trainer_params: Optional[dict] = None,
        train_dataloader_params: Optional[dict] = None,
        test_dataloader_params: Optional[dict] = None,
        val_dataloader_params: Optional[dict] = None,
        split_params: Optional[dict] = None,
    ):
        """Init generic N-BEATS model.

        Parameters
        ----------
        input_size:
            Input data size.
        output_size:
            Forecast size.
        loss:
            Optimization objective.
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
        train_batch_size:
            batch size for training
        test_batch_size:
            batch size for testing
        optimizer_params:
            parameters for optimizer for Adam optimizer (api reference :py:class:`torch.optim.Adam`)
        trainer_params:
            Pytorch ligthning  trainer parameters (api reference :py:class:`pytorch_lightning.trainer.trainer.Trainer`)
        train_dataloader_params:
            parameters for train dataloader like sampler for example (api reference :py:class:`torch.utils.data.DataLoader`)
        test_dataloader_params:
            parameters for test dataloader
        val_dataloader_params:
            parameters for validation dataloader
        split_params:
            dictionary with parameters for :py:func:`torch.utils.data.random_split` for train-test splitting
                * **train_size**: (*float*) value from 0 to 1 - fraction of samples to use for training

                * **generator**: (*Optional[torch.Generator]*) - generator for reproducibile train-test splitting

                * **torch_dataset_size**: (*Optional[int]*) - number of samples in dataset, in case of dataset not implementing ``__len__``
        """
        self.input_size = input_size
        self.output_size = output_size
        self.stacks = stacks
        self.layers = layers
        self.layer_size = layer_size
        self.lr = lr
        self.loss = loss
        self.optimizer_params = optimizer_params

        super().__init__(
            net=NBeatsGenericNet(
                input_size=input_size,
                output_size=output_size,
                stacks=stacks,
                layers=layers,
                layer_size=layer_size,
                lr=lr,
                loss=loss,
                optimizer_params=optimizer_params,
            ),
            train_batch_size=train_batch_size,
            test_batch_size=test_batch_size,
            train_dataloader_params=train_dataloader_params,
            test_dataloader_params=test_dataloader_params,
            val_dataloader_params=val_dataloader_params,
            trainer_params=trainer_params,
            split_params=split_params,
        )
