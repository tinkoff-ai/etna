from typing import List
from typing import Sequence
from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import call
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

from etna.models.base import DeepBaseModel
from etna.models.base import MultiSegmentModel
from etna.models.base import PerSegmentModel
from etna.models.base import PerSegmentPredictionIntervalModel
from etna.models.utils import select_prediction_size_timestamps


class DummyInnerModel:
    def fit(self, df: pd.DataFrame, regressors: List[str]) -> "DummyInnerModel":
        return self

    def predict(
        self,
        df: pd.DataFrame,
        prediction_size: int,
        prediction_interval: bool = False,
        quantiles: Sequence[float] = (0.025, 0.975),
    ) -> pd.DataFrame:
        if prediction_interval:
            y_pred = pd.DataFrame({"timestamp": df["timestamp"], "target": np.zeros(len(df))})
            for quantile in quantiles:
                y_pred[f"target_{quantile:.4g}"] = np.zeros(len(df))
        else:
            y_pred = np.zeros(len(df))

        y_pred = select_prediction_size_timestamps(
            prediction=y_pred, timestamp=df["timestamp"], prediction_size=prediction_size
        )
        return y_pred

    def get_model(self) -> int:
        return 0


class DummyPerSegmentModel(PerSegmentModel):

    context_size = 0

    def __init__(self):
        super().__init__(base_model=DummyInnerModel())


class DummyPerSegmentPredictionIntervalModel(PerSegmentPredictionIntervalModel):

    context_size = 0

    def __init__(self):
        super().__init__(base_model=DummyInnerModel())


class DummyMultiSegmentModel(MultiSegmentModel):

    context_size = 0

    def __init__(self):
        super().__init__(base_model=DummyInnerModel())


@pytest.mark.parametrize(
    "model",
    [
        DummyPerSegmentModel(),
        DummyPerSegmentPredictionIntervalModel(),
        DummyMultiSegmentModel(),
    ],
)
@patch("etna.models.base.check_prediction_size_value")
def test_forecast_check_prediction_size_called(check_func, model, example_tsds):
    check_func.return_value = 1
    model.fit(example_tsds)
    future = example_tsds.make_future(1)

    model.fit(example_tsds)
    model.forecast(future, prediction_size=1)

    check_func.assert_called_once()


@patch("etna.models.base.check_prediction_size_value")
def test_forecast_check_prediction_size_called_deep(check_func, deep_base_model_mock):
    horizon = 7
    ts = MagicMock()

    check_func.return_value = horizon
    DeepBaseModel.forecast(self=deep_base_model_mock, ts=ts, prediction_size=horizon)

    check_func.assert_called_once()


@pytest.fixture()
def deep_base_model_mock():
    model = MagicMock()
    model.train_batch_size = 32
    model.train_dataloader_params = {}
    model.val_dataloader_params = {}
    model.test_batch_size = 32
    model.trainer_params = {}
    model.split_params = {}
    model.context_size = 0
    return model


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


def test_deep_base_model_forecast_inverse_transform_call_check(deep_base_model_mock):
    horizon = 7
    ts = MagicMock()
    ts.index.__len__.return_value = horizon
    DeepBaseModel.forecast(self=deep_base_model_mock, ts=ts, prediction_size=horizon)
    ts.tsdataset_idx_slice.return_value.inverse_transform.assert_called_once()


def test_deep_base_model_forecast_loop(simple_df, deep_base_model_mock):
    horizon = 7
    ts = MagicMock()
    ts.index.__len__.return_value = horizon
    ts_after_tsdataset_idx_slice = MagicMock()

    raw_predict = {("A", "target"): np.arange(10).reshape(-1, 1), ("B", "target"): -np.arange(10).reshape(-1, 1)}
    deep_base_model_mock.raw_predict.return_value = raw_predict

    ts_after_tsdataset_idx_slice.df = simple_df.df.iloc[-horizon:]
    ts.tsdataset_idx_slice.return_value = ts_after_tsdataset_idx_slice

    future = DeepBaseModel.forecast(self=deep_base_model_mock, ts=ts, prediction_size=horizon)
    np.testing.assert_allclose(
        future.df.loc[:, pd.IndexSlice["A", "target"]], raw_predict[("A", "target")][:horizon, 0]
    )
    np.testing.assert_allclose(
        future.df.loc[:, pd.IndexSlice["B", "target"]], raw_predict[("B", "target")][:horizon, 0]
    )
    ts.tsdataset_idx_slice.return_value.inverse_transform.assert_called_once()
