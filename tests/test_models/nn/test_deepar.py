import pytest
from pytorch_forecasting.data import GroupNormalizer

from etna.datasets.tsdataset import TSDataset
from etna.metrics import MAE
from etna.models.nn import DeepARBaseEtnaModel
from etna.transforms import AddConstTransform
from etna.transforms import DateFlagsTransform
from etna.transforms import PytorchForecastingTransform


def test_fit_wrong_order_transform(weekly_period_df):
    ts = TSDataset(TSDataset.to_dataset(weekly_period_df), "D")
    add_const = AddConstTransform(in_column="target", value=1.0)
    pft = PytorchForecastingTransform(
        max_encoder_length=21,
        max_prediction_length=8,
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_reals=["target"],
        target_normalizer=GroupNormalizer(groups=["segment"]),
    )

    ts.fit_transform([pft, add_const])

    model = DeepARBaseEtnaModel(max_epochs=300, learning_rate=[0.1])
    with pytest.raises(ValueError, match="add PytorchForecastingTransform"):
        model.fit(ts)


@pytest.mark.long
@pytest.mark.parametrize("horizon", [8, 21])
def test_deepar_model_run_weekly_overfit(weekly_period_df, horizon):
    """
    Given: I have dataframe with 2 segments with weekly seasonality with known future
    When:
    Then: I get {horizon} periods per dataset as a forecast and they "the same" as past
    """

    ts_start = sorted(set(weekly_period_df.timestamp))[-horizon]
    train, test = (
        weekly_period_df[lambda x: x.timestamp < ts_start],
        weekly_period_df[lambda x: x.timestamp >= ts_start],
    )

    ts_train = TSDataset(TSDataset.to_dataset(train), "D")
    ts_test = TSDataset(TSDataset.to_dataset(test), "D")
    dft = DateFlagsTransform(day_number_in_week=True, day_number_in_month=False)
    pft = PytorchForecastingTransform(
        max_encoder_length=21,
        max_prediction_length=horizon,
        time_varying_known_reals=["time_idx"],
        time_varying_known_categoricals=["regressor_day_number_in_week"],
        time_varying_unknown_reals=["target"],
        target_normalizer=GroupNormalizer(groups=["segment"]),
    )

    ts_train.fit_transform([dft, pft])

    model = DeepARBaseEtnaModel(max_epochs=300, learning_rate=[0.1])
    ts_pred = ts_train.make_future(horizon)
    model.fit(ts_train)
    ts_pred = model.forecast(ts_pred)

    mae = MAE("macro")

    assert mae(ts_test, ts_pred) < 0.2207
