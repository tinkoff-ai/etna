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
from etna.pipeline import Pipeline
from etna.transforms.math import LagTransform
from etna.transforms.timestamp import DateFlagsTransform


@pytest.fixture
def ts_with_categoricals(random_seed) -> TSDataset:
    periods = 100
    df1 = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=periods)})
    df1["segment"] = "segment_1"
    df1["target"] = np.random.uniform(10, 20, size=periods)

    df2 = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=periods)})
    df2["segment"] = "segment_2"
    df2["target"] = np.random.uniform(-15, 5, size=periods)

    df_exog1 = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=periods * 2)})
    df_exog1["segment"] = "segment_1"
    df_exog1["cat_feature"] = "x"

    df_exog2 = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=periods * 2)})
    df_exog2["segment"] = "segment_2"
    df_exog2["cat_feature"] = "y"

    df = pd.concat([df1, df2]).reset_index(drop=True)
    df_exog = pd.concat([df_exog1, df_exog2]).reset_index(drop=True)

    ts = TSDataset(df=TSDataset.to_dataset(df), freq="D", df_exog=TSDataset.to_dataset(df_exog), known_future="all")

    return ts


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

    to_forecast = train.make_future(3)
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

    to_forecast = train.make_future(horizon)
    res = model.forecast(to_forecast)

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

    to_forecast = train.make_future(horizon)
    res = model.forecast(to_forecast)

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

    to_forecast = example_tsds.make_future(horizon)
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
def test_raise_error_on_unconvertable_features(ts_with_categoricals, model):
    """Check that SklearnModel raises error working with dataset with categorical features which can't be converted to numeric"""
    horizon = 7
    num_lags = 5
    lags = LagTransform(in_column="target", lags=[i + horizon for i in range(1, num_lags + 1)])
    dateflags = DateFlagsTransform()
    ts_with_categoricals.fit_transform([lags, dateflags])

    with pytest.raises(ValueError, match="Only convertible to numeric features are accepted!"):
        _ = model.fit(ts_with_categoricals)


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
