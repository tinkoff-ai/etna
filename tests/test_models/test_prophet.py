import pandas as pd

from etna.datasets.tsdataset import TSDataset
from etna.models import ProphetModel


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
    regressors_cap = new_format_exog.copy()
    regressors_cap.columns.set_levels(["regressor_cap"], level="feature", inplace=True)
    exog = pd.concat([regressors, regressors_cap], axis=1)

    ts = TSDataset(df, "1d", df_exog=exog, known_future=["regressor_exog", "regressor_cap"])

    model = ProphetModel()
    model.fit(ts)
    future_ts = ts.make_future(3)
    model.forecast(future_ts)
    if not future_ts.isnull().values.any():
        assert True
    else:
        assert False


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
