import random

import numpy as np
import pytest
import torch

from etna.datasets.tsdataset import TSDataset
from etna.metrics import MAE
from etna.models.nn import TFTModel
from etna.transforms import DateFlagsTransform
from etna.transforms import PytorchForecastingTransform


@pytest.mark.long
@pytest.mark.parametrize("horizon", [8, 21])
def test_tft_model_run_weekly_overfit(weekly_period_df, horizon):
    """
    Given: I have dataframe with 2 segments with weekly seasonality with known future
    When:
    Then: I get {horizon} periods per dataset as a forecast and they "the same" as past
    """

    SEED = 121  # noqa: N806
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    ts_start = sorted(set(weekly_period_df.timestamp))[-horizon]
    train, test = (
        weekly_period_df[lambda x: x.timestamp < ts_start],
        weekly_period_df[lambda x: x.timestamp >= ts_start],
    )

    ts_train = TSDataset(TSDataset.to_dataset(train), "1d")
    ts_test = TSDataset(TSDataset.to_dataset(test), "1d")
    dft = DateFlagsTransform(day_number_in_week=True, day_number_in_month=False)
    pft = PytorchForecastingTransform(
        max_encoder_length=21,
        min_encoder_length=21,
        max_prediction_length=horizon,
        time_varying_known_reals=["time_idx"],
        time_varying_known_categoricals=["day_number_in_week"],
        time_varying_unknown_reals=["target"],
        static_categoricals=["segment"],
        target_normalizer=None,
    )

    ts_train.fit_transform([dft, pft])

    tftmodel = TFTModel(max_epochs=300, learning_rate=[0.1])
    ts_pred = ts_train.make_future(horizon)
    tftmodel.fit(ts_train)
    ts_pred = tftmodel.forecast(ts_pred)

    mae = MAE("macro")
    assert mae(ts_test, ts_pred) < 0.24
