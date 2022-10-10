import pandas as pd
import pytest
from pytorch_forecasting.data import GroupNormalizer

from etna.datasets.tsdataset import TSDataset
from etna.metrics import MAE
from etna.models.nn import DeepARModel
from etna.pipeline import Pipeline
from etna.transforms import AddConstTransform
from etna.transforms import DateFlagsTransform
from etna.transforms import PytorchForecastingTransform
from etna.transforms import StandardScalerTransform


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

    model = DeepARModel(max_epochs=300, learning_rate=[0.1])
    with pytest.raises(ValueError, match="add PytorchForecastingTransform"):
        model.fit(ts)


@pytest.mark.long_2
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
    dft = DateFlagsTransform(day_number_in_week=True, day_number_in_month=False, out_column="regressor_dateflags")
    pft = PytorchForecastingTransform(
        max_encoder_length=21,
        max_prediction_length=horizon,
        time_varying_known_reals=["time_idx"],
        time_varying_known_categoricals=["regressor_dateflags_day_number_in_week"],
        time_varying_unknown_reals=["target"],
        target_normalizer=GroupNormalizer(groups=["segment"]),
    )

    ts_train.fit_transform([dft, pft])

    model = DeepARModel(max_epochs=300, learning_rate=[0.1])
    ts_pred = ts_train.make_future(horizon)
    model.fit(ts_train)
    ts_pred = model.forecast(ts_pred)

    mae = MAE("macro")

    assert mae(ts_test, ts_pred) < 0.2207


@pytest.mark.long_2
@pytest.mark.parametrize("horizon", [8])
def test_deepar_model_run_weekly_overfit_with_scaler(ts_dataset_weekly_function_with_horizon, horizon):
    """
    Given: I have dataframe with 2 segments with weekly seasonality with known future
    When: I use scale transformations
    Then: I get {horizon} periods per dataset as a forecast and they "the same" as past
    """

    ts_train, ts_test = ts_dataset_weekly_function_with_horizon(horizon)

    std = StandardScalerTransform(in_column="target")
    dft = DateFlagsTransform(day_number_in_week=True, day_number_in_month=False, out_column="regressor_dateflags")
    pft = PytorchForecastingTransform(
        max_encoder_length=21,
        max_prediction_length=horizon,
        time_varying_known_reals=["time_idx"],
        time_varying_known_categoricals=["regressor_dateflags_day_number_in_week"],
        time_varying_unknown_reals=["target"],
        target_normalizer=GroupNormalizer(groups=["segment"]),
    )

    ts_train.fit_transform([std, dft, pft])

    model = DeepARModel(max_epochs=300, learning_rate=[0.1])
    ts_pred = ts_train.make_future(horizon)
    model.fit(ts_train)
    ts_pred = model.forecast(ts_pred)

    mae = MAE("macro")

    assert mae(ts_test, ts_pred) < 0.2207


def test_forecast_without_make_future(weekly_period_df):
    ts = TSDataset(TSDataset.to_dataset(weekly_period_df), "D")
    pft = PytorchForecastingTransform(
        max_encoder_length=21,
        min_encoder_length=21,
        max_prediction_length=8,
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_reals=["target"],
        static_categoricals=["segment"],
        target_normalizer=None,
    )

    ts.fit_transform([pft])

    model = DeepARModel(max_epochs=1)
    model.fit(ts)
    ts.df.index = ts.df.index + pd.Timedelta(days=len(ts.df))
    with pytest.raises(ValueError, match="The future is not generated!"):
        _ = model.forecast(ts=ts)


@pytest.mark.parametrize("freq", ["1M", "1D", "A-DEC", "1B", "1H"])
def test_forecast_with_different_freq(weekly_period_df, freq):
    df = TSDataset.to_dataset(weekly_period_df)
    df.index = pd.Index(pd.date_range("2021-01-01", freq=freq, periods=len(df)), name="timestamp")

    ts = TSDataset(df, freq=freq)
    horizon = 20

    transform_deepar = PytorchForecastingTransform(
        max_encoder_length=horizon,
        max_prediction_length=horizon,
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_reals=["target"],
        target_normalizer=GroupNormalizer(groups=["segment"]),
    )

    model_deepar = DeepARModel(max_epochs=2, learning_rate=[0.01], gpus=0, batch_size=64)
    pipeline_deepar = Pipeline(model=model_deepar, horizon=horizon, transforms=[transform_deepar])
    pipeline_deepar.fit(ts=ts)
    forecast = pipeline_deepar.forecast()

    assert len(forecast.df) == horizon
    assert pd.infer_freq(forecast.df.index) in {freq, freq[1:]}


def test_prediction_interval_run_infuture(example_tsds):
    horizon = 10
    transform = PytorchForecastingTransform(
        max_encoder_length=horizon,
        max_prediction_length=horizon,
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_reals=["target"],
        target_normalizer=GroupNormalizer(groups=["segment"]),
    )
    example_tsds.fit_transform([transform])
    model = DeepARModel(max_epochs=2, learning_rate=[0.01], gpus=0, batch_size=64)
    model.fit(example_tsds)
    future = example_tsds.make_future(horizon)
    forecast = model.forecast(future, prediction_interval=True, quantiles=[0.025, 0.975])
    for segment in forecast.segments:
        segment_slice = forecast[:, segment, :][segment]
        assert {"target_0.025", "target_0.975", "target"}.issubset(segment_slice.columns)
        assert (segment_slice["target_0.975"] - segment_slice["target_0.025"] >= 0).all()
        assert (segment_slice["target"] - segment_slice["target_0.025"] >= 0).all()
        assert (segment_slice["target_0.975"] - segment_slice["target"] >= 0).all()
