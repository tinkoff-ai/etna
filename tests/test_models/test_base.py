from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import call
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

from etna.datasets import TSDataset
from etna.models.base import AbstractModel
from etna.models.base import DeepBaseModel


class DummyModel(AbstractModel):
    @property
    def context_size(self) -> int:
        return 0

    def fit(self, ts: TSDataset) -> "DummyModel":
        return self

    def get_model(self) -> int:
        return 0


def test_default_params_to_tune():
    dummy = DummyModel()
    assert dummy.params_to_tune() == {}


@pytest.fixture()
def deep_base_model_mock():
    model = MagicMock()
    model.encoder_length = 10
    model.train_batch_size = 32
    model.train_dataloader_params = {}
    model.val_dataloader_params = {}
    model.test_batch_size = 32
    model.trainer_params = {}
    model.split_params = {}
    return model


@pytest.fixture()
def ts_mock():
    torch_dataset = MagicMock()
    torch_dataset.index.__len__.return_value = 100
    return torch_dataset


@pytest.fixture()
def sized_torch_dataset_mock():
    torch_dataset = MagicMock()
    torch_dataset.__len__.return_value = 100
    return torch_dataset


@pytest.fixture()
def unsized_torch_dataset_mock():
    torch_dataset = Mock()
    return torch_dataset


def test_deep_base_model_fit_calls_check():
    deep_base_model = DeepBaseModel.fit(self=MagicMock(), ts=MagicMock())
    deep_base_model.raw_fit.assert_called_once()


@pytest.mark.parametrize("loggers", [[], ["logger"]])
@patch("etna.models.base.random_split", return_value=(MagicMock(), MagicMock()))
@patch("etna.models.base.DataLoader")
@patch("etna.models.base.Trainer")
def test_deep_base_model_raw_fit(
    trainer, dataloader, random_split, deep_base_model_mock, sized_torch_dataset_mock, loggers
):
    deep_base_model_mock.trainer_params = {"logger": loggers}
    DeepBaseModel.raw_fit(self=deep_base_model_mock, torch_dataset=sized_torch_dataset_mock)
    trainer.assert_called_with(logger=loggers)
    trainer.return_value.fit.assert_called_with(
        deep_base_model_mock.net, train_dataloaders=dataloader.return_value, val_dataloaders=None
    )
    random_split.assert_not_called()


def _test_deep_base_model_raw_fit_split_params(
    trainer,
    random_split,
    dataloader,
    deep_base_model_mock,
    torch_dataset_mock,
    torch_dataset_train_size,
    torch_dataset_val_size,
):
    DeepBaseModel.raw_fit(self=deep_base_model_mock, torch_dataset=torch_dataset_mock)
    trainer.return_value.fit.assert_called_with(
        deep_base_model_mock.net, train_dataloaders=dataloader.return_value, val_dataloaders=dataloader.return_value
    )
    calls = [
        call(random_split.return_value[0], batch_size=deep_base_model_mock.train_batch_size, shuffle=True),
        call(random_split.return_value[1], batch_size=deep_base_model_mock.train_batch_size, shuffle=False),
    ]
    dataloader.assert_has_calls(calls)
    random_split.assert_called_with(
        torch_dataset_mock,
        lengths=[torch_dataset_train_size, torch_dataset_val_size],
        generator=None,
    )


@patch("etna.models.base.random_split", return_value=(MagicMock(), MagicMock()))
@patch("etna.models.base.DataLoader")
@patch("etna.models.base.Trainer")
def test_deep_base_model_raw_fit_split_params_with_sized_torch_dataset(
    trainer, dataloader, random_split, deep_base_model_mock, sized_torch_dataset_mock
):
    torch_dataset_size = len(sized_torch_dataset_mock)
    train_size = 0.8
    torch_dataset_train_size = int(torch_dataset_size * train_size)
    torch_dataset_val_size = torch_dataset_size - torch_dataset_train_size
    deep_base_model_mock.split_params = {"train_size": train_size}
    _test_deep_base_model_raw_fit_split_params(
        trainer,
        random_split,
        dataloader,
        deep_base_model_mock,
        sized_torch_dataset_mock,
        torch_dataset_train_size,
        torch_dataset_val_size,
    )


@patch("etna.models.base.random_split", return_value=(MagicMock(), MagicMock()))
@patch("etna.models.base.DataLoader")
@patch("etna.models.base.Trainer")
def test_deep_base_model_raw_fit_split_params_with_unsized_torch_dataset(
    trainer, dataloader, random_split, deep_base_model_mock, unsized_torch_dataset_mock
):
    torch_dataset_size = 50
    train_size = 0.8
    torch_dataset_train_size = int(torch_dataset_size * train_size)
    torch_dataset_val_size = torch_dataset_size - torch_dataset_train_size
    deep_base_model_mock.split_params = {"train_size": train_size, "torch_dataset_size": torch_dataset_size}

    _test_deep_base_model_raw_fit_split_params(
        trainer,
        random_split,
        dataloader,
        deep_base_model_mock,
        unsized_torch_dataset_mock,
        torch_dataset_train_size,
        torch_dataset_val_size,
    )


@patch("etna.models.base.DataLoader")
def test_deep_base_model_raw_predict_call(dataloader, deep_base_model_mock):
    batch = {"segment": ["segment1", "segment2"], "target": torch.Tensor([[1, 2], [3, 4]])}
    dataloader.return_value = [batch]
    deep_base_model_mock.net.return_value = batch["target"]
    predictions_dict = DeepBaseModel.raw_predict(self=deep_base_model_mock, torch_dataset=MagicMock())
    deep_base_model_mock.net.eval.assert_called_once()
    np.testing.assert_allclose(predictions_dict[("segment1", "target")], batch["target"][0].numpy())
    np.testing.assert_allclose(predictions_dict[("segment2", "target")], batch["target"][1].numpy())


def test_deep_base_model_forecast_fail_not_enough_context(deep_base_model_mock, ts_mock):
    horizon = len(ts_mock.index)
    with pytest.raises(ValueError, match="Given context isn't big enough"):
        _ = DeepBaseModel.forecast(self=deep_base_model_mock, ts=ts_mock, prediction_size=horizon)


def test_deep_base_model_forecast_loop(simple_df, deep_base_model_mock, ts_mock):
    ts_after_tsdataset_idx_slice = MagicMock()
    horizon = 7

    raw_predict = {("A", "target"): np.arange(10).reshape(-1, 1), ("B", "target"): -np.arange(10).reshape(-1, 1)}
    deep_base_model_mock.raw_predict.return_value = raw_predict

    ts_after_tsdataset_idx_slice.df = simple_df.df.iloc[-horizon:]
    ts_mock.tsdataset_idx_slice.return_value = ts_after_tsdataset_idx_slice

    future = DeepBaseModel.forecast(self=deep_base_model_mock, ts=ts_mock, prediction_size=horizon)
    np.testing.assert_allclose(
        future.df.loc[:, pd.IndexSlice["A", "target"]], raw_predict[("A", "target")][:horizon, 0]
    )
    np.testing.assert_allclose(
        future.df.loc[:, pd.IndexSlice["B", "target"]], raw_predict[("B", "target")][:horizon, 0]
    )


def test_deep_base_model_forecast_throw_error_on_return_components():
    with pytest.raises(NotImplementedError, match="This mode isn't currently implemented!"):
        DeepBaseModel.forecast(self=Mock(), ts=Mock(), prediction_size=Mock(), return_components=True)
