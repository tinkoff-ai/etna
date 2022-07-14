from unittest.mock import MagicMock
from unittest.mock import patch

from etna.models.base import DeepBaseModel


def test_deep_base_model_fit_calls_check():
    deep_base_model = DeepBaseModel.fit(self=MagicMock(), ts=MagicMock())
    deep_base_model.raw_fit.assert_called_once()


@patch("etna.models.base.random_split", return_value=(MagicMock(), MagicMock()))
@patch("etna.models.base.DataLoader")
@patch("etna.models.base.Trainer")
def test_deep_base_model_raw_fit_split_params(trainer, dataloader, random_split):

    torch_dataset_size = 100
    train_size = 0.8
    self_mock = MagicMock()

    self_mock.split_params = {"train_size": train_size, "torch_dataset_size": torch_dataset_size}
    self_mock.train_batch_size = 32
    self_mock.train_dataloader_params = {}
    self_mock.val_dataloader_params = {}
    self_mock.test_batch_size = 32
    self_mock.trainer_params = {}

    torch_dataset = MagicMock()
    torch_dataset.__len__.return_value = torch_dataset_size

    DeepBaseModel.raw_fit(self=self_mock, torch_dataset=torch_dataset)
    trainer.return_value.fit.assert_called_once()
    random_split.assert_called_with(
        torch_dataset,
        lengths=[int(torch_dataset_size * train_size), torch_dataset_size - int(torch_dataset_size * train_size)],
        generator=None,
    )


@patch("etna.models.base.DataLoader")
def test_deep_base_model_raw_predict_call_eval_check(dataloader):
    self_mock = MagicMock()
    DeepBaseModel.raw_predict(self=self_mock, torch_dataset=MagicMock())
    self_mock.eval.assert_called_once()


def test_deep_base_model_forecast_inverse_transform_call_check():
    self_mock = MagicMock()
    ts = MagicMock()
    horizon = 7
    DeepBaseModel.forecast(self=self_mock, ts=ts, horizon=horizon)
    ts.tsdataset_idx_slice.return_value.inverse_transform.assert_called_once()
