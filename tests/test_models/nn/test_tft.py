import pandas as pd
import pytest

from etna.datasets.tsdataset import TSDataset
from etna.metrics import MAE
from etna.models.nn import TFTModel
from etna.transforms import AddConstTransform
from etna.transforms import DateFlagsTransform
from etna.transforms import PytorchForecastingTransform
from etna.transforms import StandardScalerTransform


def test_fit_wrong_order_transform(weekly_period_df):
    ts = TSDataset(TSDataset.to_dataset(weekly_period_df), "D")
    add_const = AddConstTransform(in_column="target", value=1.0)
    pft = PytorchForecastingTransform(
        max_encoder_length=21,
        min_encoder_length=21,
        max_prediction_length=8,
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_reals=["target"],
        static_categoricals=["segment"],
        target_normalizer=None,
    )

    ts.fit_transform([pft, add_const])

    model = TFTModel(max_epochs=300, learning_rate=[0.1])
    with pytest.raises(ValueError, match="add PytorchForecastingTransform"):
        model.fit(ts)


@pytest.mark.long_2
@pytest.mark.parametrize("horizon", [8, 21])
def test_tft_model_run_weekly_overfit(ts_dataset_weekly_function_with_horizon, horizon):
    """
    Given: I have dataframe with 2 segments with weekly seasonality with known future
    When:
    Then: I get {horizon} periods per dataset as a forecast and they "the same" as past
    """

    ts_train, ts_test = ts_dataset_weekly_function_with_horizon(horizon)

    dft = DateFlagsTransform(day_number_in_week=True, day_number_in_month=False, out_column="regressor_dateflag")
    pft = PytorchForecastingTransform(
        max_encoder_length=21,
        min_encoder_length=21,
        max_prediction_length=horizon,
        time_varying_known_reals=["time_idx"],
        time_varying_known_categoricals=["regressor_dateflag_day_number_in_week"],
        time_varying_unknown_reals=["target"],
        static_categoricals=["segment"],
        target_normalizer=None,
    )

    ts_train.fit_transform([dft, pft])

    model = TFTModel(max_epochs=300, learning_rate=[0.1])
    ts_pred = ts_train.make_future(horizon)
    model.fit(ts_train)
    ts_pred = model.forecast(ts_pred)

    mae = MAE("macro")
    assert mae(ts_test, ts_pred) < 0.24


@pytest.mark.long_2
@pytest.mark.parametrize("horizon", [8])
def test_tft_model_run_weekly_overfit_with_scaler(ts_dataset_weekly_function_with_horizon, horizon):
    """
    Given: I have dataframe with 2 segments with weekly seasonality with known future
    When: I use scale transformations
    Then: I get {horizon} periods per dataset as a forecast and they "the same" as past
    """

    ts_train, ts_test = ts_dataset_weekly_function_with_horizon(horizon)

    std = StandardScalerTransform(in_column="target")
    dft = DateFlagsTransform(day_number_in_week=True, day_number_in_month=False, out_column="regressor_dateflag")
    pft = PytorchForecastingTransform(
        max_encoder_length=21,
        min_encoder_length=21,
        max_prediction_length=horizon,
        time_varying_known_reals=["time_idx"],
        time_varying_known_categoricals=["regressor_dateflag_day_number_in_week"],
        time_varying_unknown_reals=["target"],
        static_categoricals=["segment"],
        target_normalizer=None,
    )

    ts_train.fit_transform([std, dft, pft])

    model = TFTModel(max_epochs=300, learning_rate=[0.1])
    ts_pred = ts_train.make_future(horizon)
    model.fit(ts_train)
    ts_pred = model.forecast(ts_pred)

    mae = MAE("macro")
    assert mae(ts_test, ts_pred) < 0.24


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

    model = TFTModel(max_epochs=1)
    model.fit(ts)
    ts.df.index = ts.df.index + pd.Timedelta(days=len(ts.df))
    with pytest.raises(ValueError, match="The future is not generated!"):
        _ = model.forecast(ts=ts)


def _get_default_transform(horizon: int):
    return PytorchForecastingTransform(
        max_encoder_length=21,
        min_encoder_length=21,
        max_prediction_length=horizon,
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_reals=["target"],
        static_categoricals=["segment"],
        target_normalizer=None,
    )


def test_prediction_interval_run_infuture(example_tsds):
    horizon = 10
    transform = _get_default_transform(horizon)
    example_tsds.fit_transform([transform])
    model = TFTModel(max_epochs=8, learning_rate=[0.1], gpus=0, batch_size=64)
    model.fit(example_tsds)
    future = example_tsds.make_future(horizon)
    forecast = model.forecast(future, prediction_interval=True, quantiles=[0.02, 0.98])
    for segment in forecast.segments:
        segment_slice = forecast[:, segment, :][segment]
        assert {"target_0.02", "target_0.98", "target"}.issubset(segment_slice.columns)
        assert (segment_slice["target_0.98"] - segment_slice["target_0.02"] >= 0).all()
        assert (segment_slice["target"] - segment_slice["target_0.02"] >= 0).all()
        assert (segment_slice["target_0.98"] - segment_slice["target"] >= 0).all()


def test_prediction_interval_run_infuture_warning_not_found_quantiles(example_tsds):
    horizon = 10
    transform = _get_default_transform(horizon)
    example_tsds.fit_transform([transform])
    model = TFTModel(max_epochs=2, learning_rate=[0.1], gpus=0, batch_size=64)
    model.fit(example_tsds)
    future = example_tsds.make_future(horizon)
    with pytest.warns(UserWarning, match="Quantiles: \[0.4\] can't be computed"):
        forecast = model.forecast(future, prediction_interval=True, quantiles=[0.02, 0.4, 0.98])
    for segment in forecast.segments:
        segment_slice = forecast[:, segment, :][segment]
        assert {"target_0.02", "target_0.98", "target"}.issubset(segment_slice.columns)
        assert {"target_0.4"}.isdisjoint(segment_slice.columns)


def test_prediction_interval_run_infuture_warning_loss(example_tsds):
    from pytorch_forecasting.metrics import MAE as MAEPF

    horizon = 10
    transform = _get_default_transform(horizon)
    example_tsds.fit_transform([transform])
    model = TFTModel(max_epochs=2, learning_rate=[0.1], gpus=0, batch_size=64, loss=MAEPF())
    model.fit(example_tsds)
    future = example_tsds.make_future(horizon)
    with pytest.warns(UserWarning, match="Quantiles can't be computed"):
        forecast = model.forecast(future, prediction_interval=True, quantiles=[0.02, 0.98])
    for segment in forecast.segments:
        segment_slice = forecast[:, segment, :][segment]
        assert {"target"}.issubset(segment_slice.columns)
        assert {"target_0.02", "target_0.98"}.isdisjoint(segment_slice.columns)
