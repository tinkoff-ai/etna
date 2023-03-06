import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostRegressor

from etna.datasets import TSDataset
from etna.datasets import generate_ar_df
from etna.metrics import MAE
from etna.models import CatBoostMultiSegmentModel
from etna.models import CatBoostPerSegmentModel
from etna.pipeline import Pipeline
from etna.transforms import DateFlagsTransform
from etna.transforms import LabelEncoderTransform
from etna.transforms import OneHotEncoderTransform
from etna.transforms.math import LagTransform
from tests.test_models.utils import assert_model_equals_loaded_original


@pytest.mark.parametrize("catboostmodel", [CatBoostMultiSegmentModel, CatBoostPerSegmentModel])
def test_run(catboostmodel, new_format_df):
    df = new_format_df
    ts = TSDataset(df, "1d")

    lags = LagTransform(lags=[3, 4, 5], in_column="target")

    ts.fit_transform([lags])

    model = catboostmodel()
    model.fit(ts)
    future_ts = ts.make_future(3, transforms=[lags])
    model.forecast(future_ts)
    future_ts.inverse_transform([lags])
    if not future_ts.isnull().values.any():
        assert True
    else:
        assert False


@pytest.mark.parametrize("catboostmodel", [CatBoostMultiSegmentModel, CatBoostPerSegmentModel])
def test_run_with_reg(catboostmodel, new_format_df, new_format_exog):
    df = new_format_df
    exog = new_format_exog
    exog.columns.set_levels(["regressor_exog"], level="feature", inplace=True)

    ts = TSDataset(df, "1d", df_exog=exog, known_future="all")

    lags = LagTransform(lags=[3, 4, 5], in_column="target")
    lags_exog = LagTransform(lags=[3, 4, 5, 6], in_column="regressor_exog")
    transforms = [lags, lags_exog]
    ts.fit_transform(transforms)

    model = catboostmodel()
    model.fit(ts)
    future_ts = ts.make_future(3, transforms=transforms)
    model.forecast(future_ts)
    future_ts.inverse_transform(transforms)
    if not future_ts.isnull().values.any():
        assert True
    else:
        assert False


@pytest.fixture
def constant_ts(size=40) -> TSDataset:
    constants = [7, 50, 130, 277, 370, 513]
    segments = [constant for constant in constants for _ in range(size)]
    ts_range = list(pd.date_range("2020-01-03", freq="D", periods=size))
    df = pd.DataFrame(
        {
            "timestamp": ts_range * len(constants),
            "target": segments,
            "segment": [f"segment_{i+1}" for i in range(len(constants)) for _ in range(size)],
        }
    )
    ts = TSDataset(TSDataset.to_dataset(df), "D")
    train, test = ts.train_test_split(test_size=5)
    return train, test


def test_catboost_multi_segment_forecast(constant_ts):
    train, test = constant_ts
    horizon = len(test.df)

    lags = LagTransform(in_column="target", lags=[10, 11, 12])
    train.fit_transform([lags])
    future = train.make_future(horizon, transforms=[lags])

    model = CatBoostMultiSegmentModel()
    model.fit(train)
    forecast = model.forecast(future)
    forecast.inverse_transform([lags])

    for segment in forecast.segments:
        assert np.allclose(test[:, segment, "target"], forecast[:, segment, "target"])


def test_get_model_multi():
    etna_model = CatBoostMultiSegmentModel()
    model = etna_model.get_model()
    assert isinstance(model, CatBoostRegressor)


def test_get_model_per_segment_before_training():
    etna_model = CatBoostPerSegmentModel()
    with pytest.raises(ValueError, match="Can not get the dict with base models, the model is not fitted!"):
        _ = etna_model.get_model()


def test_get_model_per_segment_after_training(example_tsds):
    pipeline = Pipeline(model=CatBoostPerSegmentModel(), transforms=[LagTransform(in_column="target", lags=[2, 3])])
    pipeline.fit(ts=example_tsds)
    models_dict = pipeline.model.get_model()
    assert isinstance(models_dict, dict)
    for segment in example_tsds.segments:
        assert isinstance(models_dict[segment], CatBoostRegressor)


@pytest.mark.parametrize(
    "encoder",
    [
        LabelEncoderTransform(in_column="date_flag_day_number_in_month"),
        OneHotEncoderTransform(in_column="date_flag_day_number_in_month"),
    ],
)
def test_encoder_catboost(encoder):
    df = generate_ar_df(start_time="2021-01-01", periods=20, n_segments=2)
    ts = TSDataset.to_dataset(df)
    ts = TSDataset(ts, freq="D")

    transforms = [DateFlagsTransform(week_number_in_month=True, out_column="date_flag"), encoder]
    model = CatBoostMultiSegmentModel(iterations=100)
    pipeline = Pipeline(model=model, transforms=transforms, horizon=1)
    _ = pipeline.backtest(ts=ts, metrics=[MAE()], n_folds=1)


@pytest.mark.parametrize(
    "model",
    [
        CatBoostPerSegmentModel(),
        CatBoostMultiSegmentModel(),
    ],
)
def test_save_load(model, example_tsds):
    horizon = 3
    transforms = [LagTransform(in_column="target", lags=list(range(horizon, horizon + 3)))]
    assert_model_equals_loaded_original(model=model, ts=example_tsds, transforms=transforms, horizon=horizon)


@pytest.fixture
def ts_w_exog(simple_df_relevance):
    df, df_exog = simple_df_relevance
    ts = TSDataset(df=df, df_exog=df_exog, freq="D")
    return ts


def test_forecast_components_equal_predict_components(ts_w_exog):
    train, test = ts_w_exog.train_test_split(test_size=5)
    future = train.make_future(5)

    model = CatBoostPerSegmentModel(iterations=10)
    model.fit(train)

    for segment in test.columns.get_level_values("segment"):
        segment_future = future[:, segment, :].droplevel("segment", axis=1).reset_index()
        prediction_components = model._models[segment].predict_components(df=segment_future)
        forecast_components = model._models[segment].forecast_components(df=segment_future)
        pd.testing.assert_frame_equal(prediction_components, forecast_components)


def test_forecast_components_names(ts_w_exog):
    answer = {"target_component_regressor_2", "target_component_regressor_1"}

    train, test = ts_w_exog.train_test_split(test_size=5)
    future = train.make_future(5)

    model = CatBoostPerSegmentModel(iterations=10)
    model.fit(train)

    for segment in test.columns.get_level_values("segment"):
        segment_future = future[:, segment, :].droplevel("segment", axis=1).reset_index()
        components = model._models[segment].forecast_components(df=segment_future)
        assert set(components.columns) == answer


def test_per_segment_decomposition_sums_to_target(ts_w_exog):
    ts = ts_w_exog
    train, test = ts.train_test_split(test_size=5)
    future = train.make_future(5)

    model = CatBoostPerSegmentModel(iterations=10)
    model.fit(train)

    y_pred = model.forecast(future)
    for segment in test.columns.get_level_values("segment"):
        segment_future = future[:, segment, :].droplevel("segment", axis=1).reset_index()
        components = model._models[segment].forecast_components(df=segment_future)
        y_hat_pred = np.sum(components.values, axis=1)
        np.testing.assert_allclose(y_hat_pred, y_pred[:, segment, "target"].values)


def test_multi_segment_decomposition_sums_to_target(ts_w_exog):
    ts = ts_w_exog
    train, test = ts.train_test_split(test_size=5)
    future = train.make_future(5)

    model = CatBoostMultiSegmentModel(iterations=10)
    model.fit(train)

    y_pred = model.forecast(future)
    for segment in test.columns.get_level_values("segment"):
        segment_future = future[:, segment, :].droplevel("segment", axis=1).reset_index()
        components = model._base_model.forecast_components(df=segment_future)
        y_hat_pred = np.sum(components.values, axis=1)
        np.testing.assert_allclose(y_hat_pred, y_pred[:, segment, "target"].values)
