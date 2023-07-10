from typing import List
from typing import Tuple

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression

from etna.datasets.tsdataset import TSDataset
from etna.models.linear import ElasticMultiSegmentModel
from etna.models.linear import ElasticPerSegmentModel
from etna.models.linear import LinearMultiSegmentModel
from etna.models.linear import LinearPerSegmentModel
from etna.models.linear import _LinearAdapter
from etna.pipeline import Pipeline
from etna.transforms.math import LagTransform
from etna.transforms.timestamp import DateFlagsTransform
from tests.test_models.utils import assert_model_equals_loaded_original
from tests.test_models.utils import assert_prediction_components_are_present
from tests.test_models.utils import assert_sampling_is_valid


@pytest.fixture
def df_with_regressors(example_tsds) -> Tuple[pd.DataFrame, List[str]]:
    lags = LagTransform(in_column="target", lags=[7], out_column="lag")
    dflg = DateFlagsTransform(day_number_in_week=True, day_number_in_month=True, is_weekend=False, out_column="df")
    example_tsds.fit_transform([lags, dflg])
    return example_tsds.to_pandas(flatten=True).dropna(), example_tsds.regressors


def linear_segments_by_parameters(alpha_values, intercept_values):
    dates = pd.date_range(start="2020-02-01", freq="D", periods=210)
    x = np.arange(210)
    train, test = [], []
    for i in range(3):
        train.append(pd.DataFrame())
        test.append(pd.DataFrame())
        train[i]["timestamp"], test[i]["timestamp"] = dates[:-7], dates[-7:]
        train[i]["segment"], test[i]["segment"] = f"segment_{i}", f"segment_{i}"

        alpha = alpha_values[i]
        intercept = intercept_values[i]
        target = x * alpha + intercept

        train[i]["target"], test[i]["target"] = target[:-7], target[-7:]

    train_df_all = pd.concat(train, ignore_index=True)
    test_df_all = pd.concat(test, ignore_index=True)
    train_ts = TSDataset(TSDataset.to_dataset(train_df_all), "D")
    test_ts = TSDataset(TSDataset.to_dataset(test_df_all), "D")

    return train_ts, test_ts


@pytest.fixture()
def linear_segments_ts_unique(random_seed):
    """Create TSDataset that represents 3 segments with unique linear dependency on lags in each."""
    alpha_values = [np.random.rand() * 4 - 2 for _ in range(3)]
    intercept_values = [np.random.rand() * 4 + 1 for _ in range(3)]
    return linear_segments_by_parameters(alpha_values, intercept_values)


@pytest.fixture()
def linear_segments_ts_common(random_seed):
    """Create TSDataset that represents 3 segments with common linear dependency on lags in each."""
    alpha_values = [np.random.rand() * 4 - 2] * 3
    intercept_values = [np.random.rand() * 4 + 1 for _ in range(3)]
    return linear_segments_by_parameters(alpha_values, intercept_values)


@pytest.mark.parametrize("model", (LinearPerSegmentModel(), ElasticPerSegmentModel()))
def test_not_fitted(model, linear_segments_ts_unique):
    """Check exception when trying to forecast with unfitted model."""
    train, test = linear_segments_ts_unique
    lags = LagTransform(in_column="target", lags=[3, 4, 5])
    train.fit_transform([lags])

    to_forecast = train.make_future(3, transforms=[lags])
    with pytest.raises(ValueError, match="model is not fitted!"):
        model.forecast(to_forecast)


@pytest.mark.parametrize(
    "model_class, model_class_repr",
    ((LinearPerSegmentModel, "LinearPerSegmentModel"), (LinearMultiSegmentModel, "LinearMultiSegmentModel")),
)
def test_repr_linear(model_class, model_class_repr):
    """Check __repr__ method of LinearPerSegmentModel and LinearMultiSegmentModel."""
    kwargs = {"copy_X": True, "positive": True}
    kwargs_repr = "copy_X = True, positive = True"
    model = model_class(fit_intercept=True, **kwargs)
    model_repr = model.__repr__()
    true_repr = f"{model_class_repr}(fit_intercept = True, {kwargs_repr}, )"
    assert model_repr == true_repr


@pytest.mark.parametrize(
    "model_class, model_class_repr",
    ((ElasticPerSegmentModel, "ElasticPerSegmentModel"), (ElasticMultiSegmentModel, "ElasticMultiSegmentModel")),
)
def test_repr_elastic(model_class, model_class_repr):
    """Check __repr__ method of ElasticPerSegmentModel and ElasticMultiSegmentModel."""
    kwargs = {"copy_X": True, "positive": True}
    kwargs_repr = "copy_X = True, positive = True"
    model = model_class(alpha=1.0, l1_ratio=0.5, fit_intercept=True, **kwargs)
    model_repr = model.__repr__()
    true_repr = f"{model_class_repr}(alpha = 1.0, l1_ratio = 0.5, " f"fit_intercept = True, {kwargs_repr}, )"
    assert model_repr == true_repr


@pytest.mark.parametrize("model", [LinearPerSegmentModel(), ElasticPerSegmentModel()])
@pytest.mark.parametrize("num_lags", [3, 5, 10, 20, 30])
def test_model_per_segment(linear_segments_ts_unique, num_lags, model):
    """
    Given: Dataset with 3 linear segments and LinearRegression or ElasticNet model that predicts per segment
    When: Creating of lag features to target, applying it to dataset and making forecast for horizon periods
    Then: Predictions per segment is close to real values
    """
    horizon = 7
    train, test = linear_segments_ts_unique
    lags = LagTransform(in_column="target", lags=[i + horizon for i in range(1, num_lags + 1)])
    train.fit_transform([lags])
    test.fit_transform([lags])

    model.fit(train)

    to_forecast = train.make_future(horizon, transforms=[lags])
    res = model.forecast(to_forecast)
    res.inverse_transform([lags])

    for segment in res.segments:
        assert np.allclose(test[:, segment, "target"], res[:, segment, "target"], atol=1)


@pytest.mark.parametrize("model", [LinearMultiSegmentModel(), ElasticMultiSegmentModel()])
@pytest.mark.parametrize("num_lags", [3, 5, 10, 20, 30])
def test_model_multi_segment(linear_segments_ts_common, num_lags, model):
    """
    Given: Dataset with 3 linear segments and LinearRegression or ElasticNet model that predicts across all segments
    When: Creating of lag features to target, applying it to dataset and making forecast for horizon periods
    Then: Predictions per segment is close to real values
    """
    horizon = 7
    train, test = linear_segments_ts_common
    lags = LagTransform(in_column="target", lags=[i + horizon for i in range(1, num_lags + 1)])
    train.fit_transform([lags])
    test.fit_transform([lags])

    model.fit(train)

    to_forecast = train.make_future(horizon, transforms=[lags])
    res = model.forecast(to_forecast)
    res.inverse_transform([lags])

    for segment in res.segments:
        assert np.allclose(test[:, segment, "target"], res[:, segment, "target"], atol=1)


@pytest.mark.parametrize("model", [LinearPerSegmentModel()])
def test_no_warning_on_categorical_features(example_tsds, model):
    """Check that SklearnModel raises no warning working with dataset with categorical features"""
    horizon = 7
    num_lags = 5
    lags = LagTransform(in_column="target", lags=[i + horizon for i in range(1, num_lags + 1)])
    dateflags = DateFlagsTransform()
    example_tsds.fit_transform([lags, dateflags])

    with pytest.warns(None) as record:
        _ = model.fit(example_tsds)
    assert (
        len(
            [
                warning
                for warning in record
                if str(warning.message).startswith(
                    "Arrays of bytes/strings is being converted to decimal numbers if dtype='numeric'."
                )
            ]
        )
        == 0
    )

    to_forecast = example_tsds.make_future(horizon, transforms=[lags, dateflags])
    with pytest.warns(None) as record:
        _ = model.forecast(to_forecast)
    assert (
        len(
            [
                warning
                for warning in record
                if str(warning.message).startswith(
                    "Arrays of bytes/strings is being converted to decimal numbers if dtype='numeric'."
                )
            ]
        )
        == 0
    )


@pytest.mark.parametrize("model", [LinearPerSegmentModel()])
def test_raise_error_on_unconvertable_features(ts_with_non_convertable_category_regressor, model):
    """Check that SklearnModel raises error working with dataset with categorical features which can't be converted to numeric"""
    ts = ts_with_non_convertable_category_regressor
    horizon = 7
    num_lags = 5
    lags = LagTransform(in_column="target", lags=[i + horizon for i in range(1, num_lags + 1)])
    dateflags = DateFlagsTransform()
    ts.fit_transform([lags, dateflags])

    with pytest.raises(ValueError, match="Only convertible to numeric features are allowed"):
        _ = model.fit(ts)


@pytest.mark.parametrize("model", [LinearPerSegmentModel()])
def test_raise_error_on_no_features(example_tsds, model):
    ts = example_tsds

    with pytest.raises(ValueError, match="There are not features for fitting the model"):
        _ = model.fit(ts)


@pytest.mark.parametrize("model", [LinearPerSegmentModel()])
def test_prediction_with_exogs_warning(ts_with_non_regressor_exog, model):
    ts = ts_with_non_regressor_exog
    horizon = 7
    num_lags = 5
    lags = LagTransform(in_column="target", lags=[i + horizon for i in range(1, num_lags + 1)])
    dateflags = DateFlagsTransform()
    ts.fit_transform([lags, dateflags])

    with pytest.warns(UserWarning, match="This model doesn't work with exogenous features unknown in future"):
        model.fit(ts)


@pytest.mark.parametrize(
    "etna_class,expected_model_class",
    (
        (ElasticMultiSegmentModel, ElasticNet),
        (LinearMultiSegmentModel, LinearRegression),
    ),
)
def test_get_model_multi(etna_class, expected_model_class):
    """Check that get_model method returns objects of sklearn regressor."""
    etna_model = etna_class()
    model = etna_model.get_model()
    assert isinstance(model, expected_model_class)


def test_get_model_per_segment_before_training():
    """Check that get_model method throws an error if per-segment model is not fitted yet."""
    etna_model = LinearPerSegmentModel()
    with pytest.raises(ValueError, match="Can not get the dict with base models, the model is not fitted!"):
        _ = etna_model.get_model()


@pytest.mark.parametrize(
    "etna_class,expected_model_class",
    (
        (ElasticPerSegmentModel, ElasticNet),
        (LinearPerSegmentModel, LinearRegression),
    ),
)
def test_get_model_per_segment_after_training(example_tsds, etna_class, expected_model_class):
    """Check that get_model method returns dict of objects of sklearn regressor class."""
    pipeline = Pipeline(model=etna_class(), transforms=[LagTransform(in_column="target", lags=[2, 3])])
    pipeline.fit(ts=example_tsds)
    models_dict = pipeline.model.get_model()
    assert isinstance(models_dict, dict)
    for segment in example_tsds.segments:
        assert isinstance(models_dict[segment], expected_model_class)


@pytest.mark.parametrize(
    "model", [ElasticPerSegmentModel(), LinearPerSegmentModel(), ElasticMultiSegmentModel(), LinearMultiSegmentModel()]
)
def test_save_load(model, example_tsds):
    horizon = 3
    transforms = [LagTransform(in_column="target", lags=list(range(horizon, horizon + 3)))]
    assert_model_equals_loaded_original(model=model, ts=example_tsds, transforms=transforms, horizon=horizon)


@pytest.mark.parametrize("fit_intercept", (True, False))
@pytest.mark.parametrize("regressor_constructor", (LinearRegression, ElasticNet))
def test_linear_adapter_predict_components_raise_error_if_not_fitted(
    df_with_regressors, regressor_constructor, fit_intercept
):
    df, regressors = df_with_regressors
    adapter = _LinearAdapter(regressor=regressor_constructor(fit_intercept=fit_intercept))
    with pytest.raises(ValueError, match="Model is not fitted"):
        _ = adapter.predict_components(df)


@pytest.mark.parametrize(
    "fit_intercept, expected_component_names",
    [
        (
            True,
            [
                "target_component_lag_7",
                "target_component_df_day_number_in_week",
                "target_component_df_day_number_in_month",
                "target_component_intercept",
            ],
        ),
        (
            False,
            [
                "target_component_lag_7",
                "target_component_df_day_number_in_week",
                "target_component_df_day_number_in_month",
            ],
        ),
    ],
)
@pytest.mark.parametrize("regressor_constructor", (LinearRegression, ElasticNet))
def test_linear_adapter_predict_components_correct_names(
    df_with_regressors, regressor_constructor, fit_intercept, expected_component_names
):
    df, regressors = df_with_regressors
    adapter = _LinearAdapter(regressor=regressor_constructor(fit_intercept=fit_intercept))
    adapter.fit(df=df, regressors=regressors)
    target_components = adapter.predict_components(df)
    assert sorted(target_components.columns) == sorted(expected_component_names)


@pytest.mark.parametrize("fit_intercept", (True, False))
@pytest.mark.parametrize("regressor_constructor", (LinearRegression, ElasticNet))
def test_linear_adapter_predict_components_sum_up_to_target(df_with_regressors, regressor_constructor, fit_intercept):
    df, regressors = df_with_regressors
    adapter = _LinearAdapter(regressor=regressor_constructor(fit_intercept=fit_intercept))
    adapter.fit(df=df, regressors=regressors)
    target = adapter.predict(df)
    target_components = adapter.predict_components(df)
    np.testing.assert_array_almost_equal(target, target_components.sum(axis=1), decimal=10)


@pytest.mark.parametrize(
    "model", (LinearPerSegmentModel(), ElasticPerSegmentModel(), LinearMultiSegmentModel(), ElasticMultiSegmentModel())
)
def test_prediction_decomposition(example_reg_tsds, model):
    train, test = example_reg_tsds.train_test_split(test_size=10)
    assert_prediction_components_are_present(model=model, train=train, test=test)


@pytest.mark.parametrize(
    "model", [LinearPerSegmentModel(), LinearMultiSegmentModel(), ElasticPerSegmentModel(), ElasticMultiSegmentModel()]
)
def test_params_to_tune(model, example_tsds):
    ts = example_tsds
    lags = LagTransform(in_column="target", lags=[10, 11, 12])
    ts.fit_transform([lags])
    assert len(model.params_to_tune()) > 0
    assert_sampling_is_valid(model=model, ts=ts)
