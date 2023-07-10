import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.holtwinters.results import HoltWintersResultsWrapper

from etna.datasets import TSDataset
from etna.datasets import generate_const_df
from etna.metrics import MAE
from etna.models import HoltModel
from etna.models import HoltWintersModel
from etna.models import SimpleExpSmoothingModel
from etna.models.holt_winters import _HoltWintersAdapter
from etna.pipeline import Pipeline
from tests.test_models.utils import assert_model_equals_loaded_original
from tests.test_models.utils import assert_prediction_components_are_present
from tests.test_models.utils import assert_sampling_is_valid


@pytest.fixture
def const_ts():
    """Create a constant dataset with little noise."""
    rng = np.random.default_rng(42)
    df = generate_const_df(start_time="2020-01-01", periods=100, freq="D", n_segments=3, scale=5)
    df["target"] += rng.normal(loc=0, scale=0.05, size=df.shape[0])
    return TSDataset(df=TSDataset.to_dataset(df), freq="D")


@pytest.mark.parametrize(
    "model",
    [
        HoltWintersModel(),
        HoltModel(),
        SimpleExpSmoothingModel(),
    ],
)
def test_holt_winters_fit_with_exog_warning(model, example_reg_tsds):
    """Test that Holt-Winters' models fits with exog with warning."""
    ts = example_reg_tsds
    with pytest.warns(UserWarning, match="This model doesn't work with exogenous features"):
        model.fit(ts)


@pytest.mark.parametrize(
    "model",
    [
        HoltWintersModel(),
        HoltModel(),
        SimpleExpSmoothingModel(),
    ],
)
def test_holt_winters_simple(model, example_tsds):
    """Test that Holt-Winters' models make predictions in simple case."""
    horizon = 7
    model.fit(example_tsds)
    future_ts = example_tsds.make_future(future_steps=horizon)
    res = model.forecast(future_ts)
    res = res.to_pandas(flatten=True)

    assert not res.isnull().values.any()
    assert len(res) == 14


@pytest.mark.parametrize(
    "model",
    [
        HoltWintersModel(),
        HoltModel(),
        SimpleExpSmoothingModel(),
    ],
)
def test_sanity_const_df(model, const_ts):
    """Test that Holt-Winters' models works good with almost constant dataset."""
    horizon = 7
    train_ts, test_ts = const_ts.train_test_split(test_size=horizon)
    pipeline = Pipeline(model=model, horizon=horizon)
    pipeline.fit(train_ts)
    future_ts = pipeline.forecast()

    mae = MAE(mode="macro")
    mae_value = mae(y_true=test_ts, y_pred=future_ts)
    assert mae_value < 0.05


@pytest.mark.parametrize(
    "etna_model_class",
    (
        HoltModel,
        HoltWintersModel,
        SimpleExpSmoothingModel,
    ),
)
def test_get_model_before_training(etna_model_class):
    """Check that get_model method throws an error if per-segment model is not fitted yet."""
    etna_model = etna_model_class()
    with pytest.raises(ValueError, match="Can not get the dict with base models, the model is not fitted!"):
        _ = etna_model.get_model()


@pytest.mark.parametrize(
    "etna_model_class,expected_class",
    (
        (HoltModel, HoltWintersResultsWrapper),
        (HoltWintersModel, HoltWintersResultsWrapper),
        (SimpleExpSmoothingModel, HoltWintersResultsWrapper),
    ),
)
def test_get_model_after_training(example_tsds, etna_model_class, expected_class):
    """Check that get_model method returns dict of objects of SARIMAX class."""
    pipeline = Pipeline(model=etna_model_class())
    pipeline.fit(ts=example_tsds)
    models_dict = pipeline.model.get_model()
    assert isinstance(models_dict, dict)
    for segment in example_tsds.segments:
        assert isinstance(models_dict[segment], expected_class)


@pytest.mark.parametrize("model", [HoltModel(), HoltWintersModel(), SimpleExpSmoothingModel()])
def test_save_load(model, example_tsds):
    assert_model_equals_loaded_original(model=model, ts=example_tsds, transforms=[], horizon=3)


@pytest.fixture()
def multi_trend_dfs(multitrend_df):
    df = multitrend_df.copy()
    df.columns = df.columns.droplevel("segment")
    df.reset_index(inplace=True)
    df["target"] += 10 - df["target"].min()

    return df.iloc[:-9], df.iloc[-9:]


@pytest.fixture()
def seasonal_dfs():
    target = pd.Series(
        [
            41.727458,
            24.041850,
            32.328103,
            37.328708,
            46.213153,
            29.346326,
            36.482910,
            42.977719,
            48.901525,
            31.180221,
            37.717881,
            40.420211,
            51.206863,
            31.887228,
            40.978263,
            43.772491,
            55.558567,
            33.850915,
            42.076383,
            45.642292,
            59.766780,
            35.191877,
            44.319737,
            47.913736,
        ],
        index=pd.period_range(start="2005Q1", end="2010Q4", freq="Q"),
    )

    df = pd.DataFrame(
        {
            "timestamp": target.index.to_timestamp(),
            "target": target.values,
        }
    )

    return df.iloc[:-9], df.iloc[-9:]


def test_check_mul_components_not_fitted_error():
    model = _HoltWintersAdapter()
    with pytest.raises(ValueError, match="This model is not fitted!"):
        model._check_mul_components()


def test_rescale_components_not_fitted_error():
    model = _HoltWintersAdapter()
    with pytest.raises(ValueError, match="This model is not fitted!"):
        model._rescale_components(pd.DataFrame({}))


@pytest.mark.parametrize("components_method_name", ("predict_components", "forecast_components"))
def test_decomposition_not_fitted_error(seasonal_dfs, components_method_name):
    _, test = seasonal_dfs

    model = _HoltWintersAdapter()
    components_method = getattr(model, components_method_name)

    with pytest.raises(ValueError, match="This model is not fitted!"):
        components_method(df=test)


@pytest.mark.parametrize(
    "components_method_name,use_future", (("predict_components", False), ("forecast_components", True))
)
@pytest.mark.parametrize("trend,seasonal", (("mul", "mul"), ("mul", None), (None, "mul")))
def test_check_mul_components(seasonal_dfs, trend, seasonal, components_method_name, use_future):
    train, test = seasonal_dfs

    model = _HoltWintersAdapter(trend=trend, seasonal=seasonal)
    model.fit(train, [])

    components_method = getattr(model, components_method_name)
    pred_df = test if use_future else train

    with pytest.raises(NotImplementedError, match="Forecast decomposition is only supported for additive components!"):
        components_method(df=pred_df)


@pytest.mark.parametrize(
    "components_method_name,use_future", (("predict_components", False), ("forecast_components", True))
)
@pytest.mark.parametrize("trend,trend_component", (("add", ["target_component_trend"]), (None, [])))
@pytest.mark.parametrize("seasonal,seasonal_component", (("add", ["target_component_seasonality"]), (None, [])))
def test_components_names(
    seasonal_dfs, trend, trend_component, seasonal, seasonal_component, components_method_name, use_future
):
    expected_components_names = set(trend_component + seasonal_component + ["target_component_level"])
    train, test = seasonal_dfs

    model = _HoltWintersAdapter(trend=trend, seasonal=seasonal)
    model.fit(train, [])

    components_method = getattr(model, components_method_name)

    pred_df = test if use_future else train
    components = components_method(df=pred_df)

    assert set(components.columns) == expected_components_names


@pytest.mark.parametrize(
    "components_method_name,in_sample", (("predict_components", True), ("forecast_components", False))
)
@pytest.mark.parametrize("df_names", ("seasonal_dfs", "multi_trend_dfs"))
@pytest.mark.parametrize("trend,damped_trend", (("add", True), ("add", False), (None, False)))
@pytest.mark.parametrize("seasonal", ("add", None))
@pytest.mark.parametrize("use_boxcox", (True, False))
def test_components_sum_up_to_target(
    df_names, trend, seasonal, damped_trend, use_boxcox, components_method_name, in_sample, request
):
    dfs = request.getfixturevalue(df_names)
    train, test = dfs

    model = _HoltWintersAdapter(trend=trend, seasonal=seasonal, damped_trend=damped_trend, use_boxcox=use_boxcox)
    model.fit(train, [])

    components_method = getattr(model, components_method_name)

    pred_df = train if in_sample else test

    components = components_method(df=pred_df)
    pred = model.predict(pred_df)

    np.testing.assert_allclose(np.sum(components.values, axis=1), pred)


@pytest.mark.parametrize(
    "components_method_name,in_sample", (("predict_components", True), ("forecast_components", False))
)
@pytest.mark.parametrize(
    "df_names",
    (
        "seasonal_dfs",
        "multi_trend_dfs",
    ),
)
def test_components_of_subset_sum_up_to_target(df_names, components_method_name, in_sample, request):
    dfs = request.getfixturevalue(df_names)
    train, test = dfs

    model = _HoltWintersAdapter()
    model.fit(train, [])

    components_method = getattr(model, components_method_name)

    pred_df = train if in_sample else test
    pred_df = pred_df.iloc[4:-2]

    pred = model.predict(pred_df)
    components = components_method(df=pred_df)

    np.testing.assert_allclose(np.sum(components.values, axis=1), pred)


def test_forecast_decompose_timestamp_error(seasonal_dfs):
    train, _ = seasonal_dfs

    model = _HoltWintersAdapter()
    model.fit(train, [])

    with pytest.raises(NotImplementedError, match="This model can't make forecast decomposition on history data."):
        model.forecast_components(df=train)


@pytest.mark.parametrize(
    "train_slice,decompose_slice",
    (
        (slice(None, 5), slice(5, None)),
        (slice(2, 7), slice(None, 5)),
    ),
)
def test_predict_decompose_timestamp_error(outliers_df, train_slice, decompose_slice):

    model = _HoltWintersAdapter()
    model.fit(outliers_df.iloc[train_slice], [])

    with pytest.raises(
        NotImplementedError, match="This model can't make prediction decomposition on future out-of-sample data"
    ):
        model.predict_components(df=outliers_df.iloc[decompose_slice])


@pytest.mark.parametrize("model", (SimpleExpSmoothingModel(), HoltModel(), HoltWintersModel()))
def test_prediction_decomposition(outliers_tsds, model):
    train, test = outliers_tsds.train_test_split(test_size=10)
    assert_prediction_components_are_present(model=model, train=train, test=test)


@pytest.mark.parametrize(
    "model, expected_length",
    [
        (HoltWintersModel(), 3),
        (HoltWintersModel(seasonal="add"), 4),
        (HoltModel(), 2),
        (SimpleExpSmoothingModel(), 0),
    ],
)
def test_params_to_tune(model, expected_length, const_ts):
    def skip_parameters(parameters):
        if (
            "damped_trend" in parameters
            and "trend" in parameters
            and parameters["damped_trend"]
            and parameters["trend"] is None
        ):
            return True
        return False

    ts = const_ts
    assert len(model.params_to_tune()) == expected_length
    assert_sampling_is_valid(model=model, ts=ts, skip_parameters=skip_parameters)
