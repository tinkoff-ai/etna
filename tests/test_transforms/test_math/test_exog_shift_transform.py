import numpy as np
import pandas as pd
import pytest

from etna.datasets import TSDataset
from etna.metrics import MAE
from etna.models import LinearPerSegmentModel
from etna.pipeline import Pipeline
from etna.transforms import ExogShiftTransform


def build_df_exog_with_nans(freq="D"):
    df = pd.DataFrame(
        {
            "timestamp": list(pd.date_range("2023-01-01", periods=5, freq=freq)) * 2,
            "segment": ["A"] * 5 + ["B"] * 5,
            "feat1": [1, 2, 3, 4, None] + [1, 2, 3, 4, 5],
            "feat2": [1, 2, 3, None, None] + [1, 2, 3, None, None],
            "feat3": [1, 2, 3, 4, 5] + [1, 2, 3, 4, 5],
        }
    )

    return TSDataset.to_dataset(df=df)


def build_ts_with_exogs(freq="D"):
    df = pd.DataFrame(
        {
            "timestamp": list(pd.date_range("2023-01-01", periods=4, freq=freq)) * 2,
            "segment": ["A"] * 4 + ["B"] * 4,
            "target": list(3 * np.arange(1, 5)) * 2,
        }
    )

    df = TSDataset.to_dataset(df=df)
    ts = TSDataset(df=df, df_exog=build_df_exog_with_nans(freq=freq), freq=freq)
    return ts


@pytest.fixture()
def df_exog_with_nans():
    return build_df_exog_with_nans(freq="D")


@pytest.fixture()
def ts_with_exogs():
    return build_ts_with_exogs(freq="D")


@pytest.fixture()
def ts_with_exogs_ms_freq():
    return build_ts_with_exogs(freq="MS")


@pytest.mark.parametrize(
    "expected",
    (
        {
            "feat1": pd.Timestamp("2023-01-04"),
            "feat2": pd.Timestamp("2023-01-03"),
            "feat3": pd.Timestamp("2023-01-05 00:00:00"),
        },
    ),
)
def test_save_exog_last_date(df_exog_with_nans, expected):
    t = ExogShiftTransform(lag=1)
    t._save_exog_last_date(df_exog=df_exog_with_nans)
    assert t._exog_last_date == expected


def test_negative_lag():
    with pytest.raises(ValueError, match=".* works only with positive lags"):
        ExogShiftTransform(lag=-1)


def test_horizon_not_set():
    with pytest.raises(ValueError, match="`horizon` should be specified"):
        ExogShiftTransform(lag="auto")


def test_negative_horizon_set():
    with pytest.raises(ValueError, match=".* works only with positive horizon"):
        ExogShiftTransform(lag="auto", horizon=-1)


def test_regressors_info_not_fit():
    with pytest.raises(ValueError, match="Fit the transform"):
        ExogShiftTransform(lag=1).get_regressors_info()


def test_estimate_shift_not_fit():
    with pytest.raises(ValueError, match=".* method before estimating exog shifts!"):
        ExogShiftTransform(lag="auto", horizon=1)._estimate_shift(None, None)


def test_transform_not_fit(ts_with_exogs):
    t = ExogShiftTransform(lag=1)
    with pytest.raises(ValueError, match="Transform is not fitted!"):
        t.transform(ts=ts_with_exogs)


def test_get_feature_names(example_reg_tsds, expected={"regressor_exog_weekend"}):
    t = ExogShiftTransform(lag=1)
    t.fit(ts=example_reg_tsds)
    feature_names = t._get_feature_names(example_reg_tsds.df)
    assert set(feature_names) == expected


def test_get_feature_names_no_exog_error(ts_with_exogs):
    t = ExogShiftTransform(lag=1)
    t.fit(ts=ts_with_exogs)
    df = ts_with_exogs.df.drop(columns=["feat3"], level=1)

    with pytest.raises(ValueError, match="Feature `feat3` is expected to be in the dataframe!"):
        t._get_feature_names(df)


@pytest.mark.parametrize(
    "ts_name", ("toy_dataset_with_mean_shift_in_target", "product_level_constant_forecast_with_target_components")
)
def test_ts_with_quantiles_and_components(ts_name, request):
    ts = request.getfixturevalue(ts_name)
    t = ExogShiftTransform(lag=1)
    t.fit(ts=ts)
    assert t.get_regressors_info() == []


@pytest.mark.parametrize(
    "horizon,expected",
    ((1, {"feat1_shift_1", "feat2_shift_2"}), (2, {"feat1_shift_2", "feat2_shift_3", "feat3_shift_1"})),
)
def test_regressors_info(ts_with_exogs, horizon, expected):
    t = ExogShiftTransform(lag="auto", horizon=horizon)
    t.fit(ts=ts_with_exogs)
    assert set(t.get_regressors_info()) == expected


@pytest.mark.parametrize(
    "lag,horizon,expected",
    (
        (1, None, {"feat1": 1, "feat2": 1, "feat3": 1}),
        ("auto", 1, {"feat1": 1, "feat2": 2, "feat3": 0}),
        ("auto", 2, {"feat1": 2, "feat2": 3, "feat3": 1}),
    ),
)
def test_estimate_shift(ts_with_exogs, lag, horizon, expected):
    t = ExogShiftTransform(lag=lag, horizon=horizon)
    t.fit(ts=ts_with_exogs)
    assert t._exog_shifts == expected


@pytest.mark.parametrize("lag", (1, "auto"))
def test_shift_no_exog(simple_df, lag, expected={"target"}):
    t = ExogShiftTransform(lag=lag, horizon=1)
    transformed = t.fit_transform(simple_df)
    assert set(transformed.df.columns.get_level_values("feature")) == expected


@pytest.mark.parametrize(
    "lag,horizon,expected",
    (
        (1, None, {"feat1_shift_1", "feat2_shift_1", "feat3_shift_1", "target"}),
        ("auto", 1, {"feat1_shift_1", "feat2_shift_2", "feat3", "target"}),
        ("auto", 2, {"feat1_shift_2", "feat2_shift_3", "feat3_shift_1", "target"}),
    ),
)
@pytest.mark.parametrize(
    "ts_name",
    (
        "ts_with_exogs",
        "ts_with_exogs_ms_freq",
    ),
)
def test_transformed_names(ts_name, lag, horizon, expected, request):
    ts = request.getfixturevalue(ts_name)

    t = ExogShiftTransform(lag=lag, horizon=horizon)
    transformed = t.fit_transform(ts=ts)
    column_names = transformed.df.columns.get_level_values("feature")
    assert set(column_names) == expected


@pytest.mark.parametrize("lag", (3, "auto"))
@pytest.mark.parametrize("horizon", range(1, 3))
@pytest.mark.parametrize(
    "ts_name",
    (
        "ts_with_exogs",
        "ts_with_exogs_ms_freq",
    ),
)
def test_pipeline_forecast(ts_name, lag, horizon, request):
    ts = request.getfixturevalue(ts_name)

    pipeline = Pipeline(
        transforms=[ExogShiftTransform(lag=lag, horizon=horizon)], model=LinearPerSegmentModel(), horizon=horizon
    )
    pipeline.fit(ts)
    pipeline.forecast()


@pytest.mark.parametrize("lag", (3, "auto"))
@pytest.mark.parametrize("horizon", range(7, 10))
def test_pipeline_backtest(example_reg_tsds, lag, horizon):
    ts = example_reg_tsds

    pipeline = Pipeline(
        transforms=[ExogShiftTransform(lag=lag, horizon=horizon)], model=LinearPerSegmentModel(), horizon=horizon
    )

    pipeline.backtest(ts=ts, metrics=[MAE()])
