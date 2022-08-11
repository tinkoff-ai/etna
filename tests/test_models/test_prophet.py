import numpy as np
import pandas as pd
import pytest
from prophet import Prophet

from etna.datasets.tsdataset import TSDataset
from etna.models import ProphetModel
from etna.pipeline import Pipeline


def test_run(new_format_df):
    df = new_format_df

    ts = TSDataset(df, "1d")

    model = ProphetModel()
    model.fit(ts)
    future_ts = ts.make_future(3)
    model.forecast(future_ts)
    if not future_ts.isnull().values.any():
        assert True
    else:
        assert False


def test_run_with_reg(new_format_df, new_format_exog):
    df = new_format_df

    regressors = new_format_exog.copy()
    regressors.columns.set_levels(["regressor_exog"], level="feature", inplace=True)
    regressors_floor = new_format_exog.copy()
    regressors_floor.columns.set_levels(["floor"], level="feature", inplace=True)
    regressors_cap = regressors_floor.copy() + 1
    regressors_cap.columns.set_levels(["cap"], level="feature", inplace=True)
    exog = pd.concat([regressors, regressors_floor, regressors_cap], axis=1)

    ts = TSDataset(df, "1d", df_exog=exog, known_future="all")

    model = ProphetModel(growth="logistic")
    model.fit(ts)
    future_ts = ts.make_future(3)
    model.forecast(future_ts)
    if not future_ts.isnull().values.any():
        assert True
    else:
        assert False


def test_run_with_cap_floor():
    cap = 101
    floor = -1

    df = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2020-01-01", periods=100),
            "segment": "segment_0",
            "target": list(range(100)),
        }
    )
    df_exog = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2020-01-01", periods=120),
            "segment": "segment_0",
            "cap": cap,
            "floor": floor,
        }
    )
    ts = TSDataset(df=TSDataset.to_dataset(df), df_exog=TSDataset.to_dataset(df_exog), freq="D", known_future="all")

    model = ProphetModel(growth="logistic")
    pipeline = Pipeline(model=model, horizon=7)
    pipeline.fit(ts)

    ts_future = pipeline.forecast()
    df_future = ts_future.to_pandas(flatten=True)

    assert np.all(df_future["target"] < cap)


def test_prediction_interval_run_insample(example_tsds):
    model = ProphetModel()
    model.fit(example_tsds)
    forecast = model.forecast(example_tsds, prediction_interval=True, quantiles=[0.025, 0.975])
    for segment in forecast.segments:
        segment_slice = forecast[:, segment, :][segment]
        assert {"target_0.025", "target_0.975", "target"}.issubset(segment_slice.columns)
        assert (segment_slice["target_0.975"] - segment_slice["target_0.025"] >= 0).all()


def test_prediction_interval_run_infuture(example_tsds):
    model = ProphetModel()
    model.fit(example_tsds)
    future = example_tsds.make_future(10)
    forecast = model.forecast(future, prediction_interval=True, quantiles=[0.025, 0.975])
    for segment in forecast.segments:
        segment_slice = forecast[:, segment, :][segment]
        assert {"target_0.025", "target_0.975", "target"}.issubset(segment_slice.columns)
        assert (segment_slice["target_0.975"] - segment_slice["target_0.025"] >= 0).all()


def test_prophet_save_regressors_on_fit(example_reg_tsds):
    model = ProphetModel()
    model.fit(ts=example_reg_tsds)
    for segment_model in model._models.values():
        assert sorted(segment_model.regressor_columns) == example_reg_tsds.regressors


def test_get_model_before_training():
    """Check that get_model method throws an error if per-segment model is not fitted yet."""
    etna_model = ProphetModel()
    with pytest.raises(ValueError, match="Can not get the dict with base models, the model is not fitted!"):
        _ = etna_model.get_model()


def test_get_model_after_training(example_tsds):
    """Check that get_model method returns dict of objects of Prophet class."""
    pipeline = Pipeline(model=ProphetModel())
    pipeline.fit(ts=example_tsds)
    models_dict = pipeline.model.get_model()
    assert isinstance(models_dict, dict)
    for segment in example_tsds.segments:
        assert isinstance(models_dict[segment], Prophet)
