import pathlib
from unittest.mock import Mock
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from etna.datasets import TSDataset
from etna.datasets.utils import match_target_quantiles
from etna.distributions import CategoricalDistribution
from etna.distributions import FloatDistribution
from etna.distributions import IntDistribution
from etna.metrics import MAE
from etna.metrics import Coverage
from etna.metrics import Width
from etna.models import CatBoostMultiSegmentModel
from etna.models import LinearPerSegmentModel
from etna.models import NaiveModel
from etna.models import ProphetModel
from etna.pipeline.hierarchical_pipeline import HierarchicalPipeline
from etna.reconciliation import BottomUpReconciliator
from etna.reconciliation import TopDownReconciliator
from etna.transforms import DateFlagsTransform
from etna.transforms import LagTransform
from etna.transforms import LinearTrendTransform
from etna.transforms import MeanTransform
from tests.test_pipeline.utils import assert_pipeline_equals_loaded_original
from tests.test_pipeline.utils import assert_pipeline_forecasts_given_ts
from tests.test_pipeline.utils import assert_pipeline_forecasts_given_ts_with_prediction_intervals


@pytest.mark.parametrize(
    "reconciliator",
    (
        TopDownReconciliator(target_level="a", source_level="b", period=1, method="AHP"),
        BottomUpReconciliator(target_level="a", source_level="b"),
    ),
)
def test_init_pass(reconciliator):
    model = NaiveModel()
    pipeline = HierarchicalPipeline(reconciliator=reconciliator, model=model, transforms=[], horizon=1)
    assert isinstance(pipeline.reconciliator, type(reconciliator))


@pytest.mark.parametrize(
    "reconciliator",
    (
        TopDownReconciliator(target_level="market", source_level="total", period=1, method="AHP"),
        BottomUpReconciliator(target_level="total", source_level="market"),
    ),
)
def test_fit_mapping_matrix(market_level_simple_hierarchical_ts, reconciliator):
    model = NaiveModel()
    pipeline = HierarchicalPipeline(reconciliator=reconciliator, model=model, transforms=[], horizon=1)

    pipeline.reconciliator.fit = Mock()
    pipeline.fit(market_level_simple_hierarchical_ts)
    pipeline.reconciliator.fit.assert_called()


@pytest.mark.parametrize(
    "reconciliator",
    (
        TopDownReconciliator(target_level="market", source_level="total", period=1, method="AHP"),
        BottomUpReconciliator(target_level="total", source_level="market"),
    ),
)
def test_fit_dataset_level(market_level_simple_hierarchical_ts, reconciliator):
    model = NaiveModel()
    pipeline = HierarchicalPipeline(reconciliator=reconciliator, model=model, transforms=[], horizon=1)
    pipeline.fit(market_level_simple_hierarchical_ts)
    assert pipeline.ts.current_df_level == reconciliator.source_level


def test_fit_no_hierarchy(simple_no_hierarchy_ts):
    model = NaiveModel()
    reconciliator = BottomUpReconciliator(target_level="total", source_level="market")
    pipeline = HierarchicalPipeline(reconciliator=reconciliator, model=model, transforms=[], horizon=1)
    with pytest.raises(ValueError, match="The method can be applied only to instances with a hierarchy!"):
        pipeline.fit(simple_no_hierarchy_ts)


@pytest.mark.parametrize(
    "reconciliator,answer",
    (
        (TopDownReconciliator(target_level="market", source_level="total", period=1, method="AHP"), 10),
        (BottomUpReconciliator(target_level="total", source_level="market"), np.array([[3, 7]])),
    ),
)
def test_raw_forecast_correctness(market_level_constant_hierarchical_ts, reconciliator, answer):
    model = NaiveModel()
    pipeline = HierarchicalPipeline(reconciliator=reconciliator, model=model, transforms=[], horizon=1)
    pipeline.fit(ts=market_level_constant_hierarchical_ts)
    forecast = pipeline.raw_forecast(ts=market_level_constant_hierarchical_ts)
    np.testing.assert_allclose(forecast[..., "target"].values, answer)


@pytest.mark.parametrize(
    "reconciliator,answer",
    (
        (TopDownReconciliator(target_level="market", source_level="total", period=1, method="AHP"), 10),
        (BottomUpReconciliator(target_level="total", source_level="market"), np.array([[3, 7]])),
    ),
)
def test_raw_predict_correctness(market_level_constant_hierarchical_ts, reconciliator, answer):
    ts = market_level_constant_hierarchical_ts

    pipeline = HierarchicalPipeline(reconciliator=reconciliator, model=NaiveModel(), transforms=[], horizon=1)
    pipeline.fit(ts=ts)
    forecast = pipeline.raw_predict(ts=ts, start_timestamp=ts.index[3])
    np.testing.assert_allclose(forecast[..., "target"].values, answer)


@pytest.mark.parametrize(
    "reconciliator",
    (
        TopDownReconciliator(target_level="market", source_level="total", period=1, method="AHP"),
        BottomUpReconciliator(target_level="total", source_level="market"),
    ),
)
def test_raw_forecast_level(market_level_simple_hierarchical_ts, reconciliator):
    model = NaiveModel()
    pipeline = HierarchicalPipeline(reconciliator=reconciliator, model=model, transforms=[], horizon=1)
    pipeline.fit(ts=market_level_simple_hierarchical_ts)
    forecast = pipeline.raw_forecast(ts=market_level_simple_hierarchical_ts)
    assert forecast.current_df_level == pipeline.reconciliator.source_level


@pytest.mark.parametrize(
    "reconciliator",
    (
        TopDownReconciliator(target_level="market", source_level="total", period=1, method="AHP"),
        BottomUpReconciliator(target_level="total", source_level="market"),
    ),
)
def test_raw_predict_level(market_level_simple_hierarchical_ts, reconciliator):
    ts = market_level_simple_hierarchical_ts

    pipeline = HierarchicalPipeline(reconciliator=reconciliator, model=NaiveModel(), transforms=[], horizon=1)
    pipeline.fit(ts=ts)
    forecast = pipeline.raw_predict(ts=ts, start_timestamp=ts.index[1])
    assert forecast.current_df_level == pipeline.reconciliator.source_level


@pytest.mark.parametrize(
    "reconciliator,answer",
    (
        (TopDownReconciliator(target_level="market", source_level="total", period=1, method="AHP"), np.array([[3, 7]])),
        (BottomUpReconciliator(target_level="total", source_level="market"), 10),
    ),
)
def test_forecast_correctness(market_level_constant_hierarchical_ts, reconciliator, answer):
    model = NaiveModel()
    pipeline = HierarchicalPipeline(reconciliator=reconciliator, model=model, transforms=[], horizon=1)
    pipeline.fit(ts=market_level_constant_hierarchical_ts)
    forecast = pipeline.forecast()
    np.testing.assert_allclose(forecast[..., "target"].values, answer)


@pytest.mark.parametrize(
    "reconciliator,answer",
    (
        (TopDownReconciliator(target_level="market", source_level="total", period=1, method="AHP"), np.array([[3, 7]])),
        (BottomUpReconciliator(target_level="total", source_level="market"), 10),
    ),
)
def test_predict_correctness(market_level_constant_hierarchical_ts, reconciliator, answer):
    ts = market_level_constant_hierarchical_ts

    pipeline = HierarchicalPipeline(reconciliator=reconciliator, model=NaiveModel(), transforms=[], horizon=1)
    pipeline.fit(ts=ts)
    forecast = pipeline.predict(ts=ts, start_timestamp=ts.index[3])
    np.testing.assert_allclose(forecast[..., "target"].values, answer)


@pytest.mark.parametrize(
    "reconciliator",
    (
        TopDownReconciliator(target_level="market", source_level="total", period=1, method="AHP"),
        BottomUpReconciliator(target_level="total", source_level="market"),
    ),
)
def test_forecast_level(market_level_simple_hierarchical_ts, reconciliator):
    model = NaiveModel()
    pipeline = HierarchicalPipeline(reconciliator=reconciliator, model=model, transforms=[], horizon=1)
    pipeline.fit(ts=market_level_simple_hierarchical_ts)
    forecast = pipeline.forecast()
    assert forecast.current_df_level == pipeline.reconciliator.target_level


@pytest.mark.parametrize(
    "reconciliator",
    (
        TopDownReconciliator(target_level="market", source_level="total", period=1, method="AHP"),
        BottomUpReconciliator(target_level="total", source_level="market"),
    ),
)
def test_predict_level(market_level_simple_hierarchical_ts, reconciliator):
    ts = market_level_simple_hierarchical_ts

    pipeline = HierarchicalPipeline(reconciliator=reconciliator, model=NaiveModel(), transforms=[], horizon=1)
    pipeline.fit(ts=ts)
    forecast = pipeline.predict(ts=ts, start_timestamp=ts.index[1])
    assert forecast.current_df_level == pipeline.reconciliator.target_level


@pytest.mark.parametrize(
    "reconciliator",
    (
        TopDownReconciliator(target_level="market", source_level="total", period=1, method="AHP"),
        BottomUpReconciliator(target_level="total", source_level="market"),
    ),
)
def test_forecast_columns_duplicates(market_level_constant_hierarchical_ts_w_exog, reconciliator):
    ts = market_level_constant_hierarchical_ts_w_exog
    model = NaiveModel()
    pipeline = HierarchicalPipeline(reconciliator=reconciliator, model=model, transforms=[], horizon=1)
    pipeline.fit(ts=ts)
    forecast = pipeline.forecast()
    assert not any(forecast.df.columns.duplicated())


@pytest.mark.parametrize(
    "reconciliator",
    (
        TopDownReconciliator(target_level="market", source_level="total", period=1, method="AHP"),
        BottomUpReconciliator(target_level="total", source_level="market"),
    ),
)
def test_predict_columns_duplicates(market_level_constant_hierarchical_ts_w_exog, reconciliator):
    ts = market_level_constant_hierarchical_ts_w_exog
    pipeline = HierarchicalPipeline(reconciliator=reconciliator, model=NaiveModel(), transforms=[], horizon=1)
    pipeline.fit(ts=ts)
    forecast = pipeline.predict(ts=ts, start_timestamp=ts.index[3])
    assert not any(forecast.df.columns.duplicated())


@pytest.mark.parametrize(
    "reconciliator",
    (
        TopDownReconciliator(target_level="market", source_level="total", period=1, method="AHP"),
        TopDownReconciliator(target_level="market", source_level="total", period=1, method="PHA"),
        BottomUpReconciliator(target_level="total", source_level="market"),
    ),
)
def test_backtest(market_level_constant_hierarchical_ts, reconciliator):
    ts = market_level_constant_hierarchical_ts
    model = NaiveModel()
    pipeline = HierarchicalPipeline(reconciliator=reconciliator, model=model, transforms=[], horizon=1)
    metrics, _, _ = pipeline.backtest(ts=ts, metrics=[MAE()], n_folds=2, aggregate_metrics=True)
    np.testing.assert_allclose(metrics["MAE"], 0)


@pytest.mark.parametrize(
    "reconciliator",
    (
        TopDownReconciliator(target_level="market", source_level="total", period=1, method="AHP"),
        TopDownReconciliator(target_level="market", source_level="total", period=1, method="PHA"),
        BottomUpReconciliator(target_level="total", source_level="market"),
    ),
)
def test_backtest_w_transforms(market_level_constant_hierarchical_ts, reconciliator):
    ts = market_level_constant_hierarchical_ts
    model = LinearPerSegmentModel()
    transforms = [
        MeanTransform(in_column="target", window=2),
        LinearTrendTransform(in_column="target"),
        LagTransform(in_column="target", lags=[1]),
    ]
    pipeline = HierarchicalPipeline(reconciliator=reconciliator, model=model, transforms=transforms, horizon=1)
    metrics, _, _ = pipeline.backtest(ts=ts, metrics=[MAE()], n_folds=2, aggregate_metrics=True)
    np.testing.assert_allclose(metrics["MAE"], 0)


@pytest.mark.parametrize(
    "reconciliator",
    (
        TopDownReconciliator(target_level="product", source_level="market", period=1, method="AHP"),
        TopDownReconciliator(target_level="product", source_level="market", period=1, method="PHA"),
        BottomUpReconciliator(target_level="total", source_level="market"),
    ),
)
def test_backtest_w_exog(product_level_constant_hierarchical_ts_with_exog, reconciliator):
    ts = product_level_constant_hierarchical_ts_with_exog
    model = LinearPerSegmentModel()
    pipeline = HierarchicalPipeline(reconciliator=reconciliator, model=model, transforms=[], horizon=1)
    metrics, _, _ = pipeline.backtest(ts=ts, metrics=[MAE()], n_folds=2, aggregate_metrics=True)
    np.testing.assert_allclose(metrics["MAE"], 0)


@pytest.mark.parametrize(
    "reconciliator",
    (
        TopDownReconciliator(target_level="product", source_level="market", period=1, method="AHP"),
        TopDownReconciliator(target_level="product", source_level="market", period=1, method="PHA"),
        BottomUpReconciliator(target_level="total", source_level="market"),
    ),
)
def test_private_forecast_prediction_interval_no_swap_after_error(
    product_level_constant_hierarchical_ts_with_exog, reconciliator
):
    ts = product_level_constant_hierarchical_ts_with_exog
    model = LinearPerSegmentModel()
    pipeline = HierarchicalPipeline(reconciliator=reconciliator, model=model, transforms=[], horizon=1)
    pipeline.backtest = Mock(side_effect=ValueError("Some error"))
    forecast_method = pipeline.forecast
    raw_forecast_method = pipeline.raw_forecast

    pipeline.fit(ts=ts)
    with pytest.raises(ValueError, match="Some error"):
        _ = pipeline.forecast(prediction_interval=True, n_folds=1, quantiles=[0.025, 0.5, 0.975])

    # check that methods aren't swapped
    assert pipeline.forecast == forecast_method
    assert pipeline.raw_forecast == raw_forecast_method


@pytest.mark.parametrize(
    "reconciliator",
    (
        TopDownReconciliator(target_level="product", source_level="market", period=1, method="PHA"),
        BottomUpReconciliator(target_level="market", source_level="product"),
    ),
)
def test_forecast_interval_presented(product_level_constant_hierarchical_ts, reconciliator):
    ts = product_level_constant_hierarchical_ts
    model = NaiveModel()
    pipeline = HierarchicalPipeline(reconciliator=reconciliator, model=model, transforms=[], horizon=2)

    pipeline.fit(ts=ts)
    forecast = pipeline.forecast(prediction_interval=True, n_folds=1, quantiles=[0.025, 0.5, 0.975])
    quantiles = match_target_quantiles(set(forecast.columns.get_level_values(1)))
    assert quantiles == {"target_0.025", "target_0.5", "target_0.975"}


@pytest.mark.parametrize(
    "reconciliator",
    (
        TopDownReconciliator(target_level="product", source_level="market", period=1, method="PHA"),
        BottomUpReconciliator(target_level="market", source_level="product"),
    ),
)
def test_predict_interval_presented(product_level_constant_hierarchical_ts, reconciliator):
    ts = product_level_constant_hierarchical_ts
    pipeline = HierarchicalPipeline(reconciliator=reconciliator, model=ProphetModel(), transforms=[], horizon=2)

    pipeline.fit(ts=ts)
    forecast = pipeline.predict(
        ts=ts, start_timestamp=ts.index[1], prediction_interval=True, quantiles=[0.025, 0.5, 0.975]
    )
    quantiles = match_target_quantiles(set(forecast.columns.get_level_values(1)))
    assert quantiles == {"target_0.025", "target_0.5", "target_0.975"}


@pytest.mark.parametrize(
    "reconciliator",
    (
        TopDownReconciliator(target_level="product", source_level="market", period=1, method="AHP"),
        TopDownReconciliator(target_level="product", source_level="market", period=1, method="PHA"),
        BottomUpReconciliator(target_level="market", source_level="product"),
        BottomUpReconciliator(target_level="total", source_level="market"),
    ),
)
def test_forecast_prediction_intervals(product_level_constant_hierarchical_ts, reconciliator):
    ts = product_level_constant_hierarchical_ts
    model = NaiveModel()
    pipeline = HierarchicalPipeline(reconciliator=reconciliator, model=model, transforms=[], horizon=2)

    pipeline.fit(ts=ts)
    forecast = pipeline.forecast(prediction_interval=True, n_folds=1)
    for segment in forecast.segments:
        target = forecast[:, segment, "target"]
        np.testing.assert_allclose(target, forecast[:, segment, "target_0.025"])
        np.testing.assert_allclose(target, forecast[:, segment, "target_0.975"])


@pytest.mark.parametrize(
    "reconciliator",
    (
        TopDownReconciliator(target_level="product", source_level="market", period=1, method="AHP"),
        TopDownReconciliator(target_level="product", source_level="market", period=1, method="PHA"),
        BottomUpReconciliator(target_level="market", source_level="product"),
        BottomUpReconciliator(target_level="total", source_level="market"),
    ),
)
def test_predict_confidence_intervals(product_level_constant_hierarchical_ts, reconciliator):
    ts = product_level_constant_hierarchical_ts
    model = ProphetModel()
    pipeline = HierarchicalPipeline(reconciliator=reconciliator, model=model, transforms=[], horizon=2)

    pipeline.fit(ts=ts)
    forecast = pipeline.predict(ts=ts, start_timestamp=ts.index[1], prediction_interval=True)
    for segment in forecast.segments:
        target = forecast[:, segment, "target"]
        np.testing.assert_allclose(target, forecast[:, segment, "target_0.025"])
        np.testing.assert_allclose(target, forecast[:, segment, "target_0.975"])


@pytest.mark.parametrize(
    "metric_type,reconciliator,answer",
    (
        (Width, TopDownReconciliator(target_level="product", source_level="market", period=1, method="AHP"), 0),
        (Width, BottomUpReconciliator(target_level="total", source_level="market"), 0),
        (Coverage, TopDownReconciliator(target_level="product", source_level="market", period=1, method="AHP"), 1),
        (Coverage, BottomUpReconciliator(target_level="total", source_level="market"), 1),
    ),
)
def test_interval_metrics(product_level_constant_hierarchical_ts, metric_type, reconciliator, answer):
    ts = product_level_constant_hierarchical_ts
    model = NaiveModel()
    pipeline = HierarchicalPipeline(reconciliator=reconciliator, model=model, transforms=[], horizon=1)

    metric = metric_type()
    results, _, _ = pipeline.backtest(
        ts=ts,
        metrics=[metric],
        n_folds=1,
        aggregate_metrics=True,
        forecast_params={"prediction_interval": True, "n_folds": 2},
    )
    np.testing.assert_allclose(results[metric.name], answer)


@patch("etna.pipeline.pipeline.Pipeline.save")
def test_save(save_mock, product_level_constant_hierarchical_ts, tmp_path):
    ts = product_level_constant_hierarchical_ts
    model = NaiveModel()
    reconciliator = BottomUpReconciliator(target_level="market", source_level="product")
    pipeline = HierarchicalPipeline(reconciliator=reconciliator, model=model, transforms=[], horizon=1)
    dir_path = pathlib.Path(tmp_path)
    path = dir_path / "dummy.zip"
    pipeline.fit(ts)

    def check_no_fit_ts(path):
        assert not hasattr(pipeline, "_fit_ts")

    save_mock.side_effect = check_no_fit_ts

    pipeline.save(path)

    save_mock.assert_called_once_with(path=path)
    assert hasattr(pipeline, "_fit_ts")


@patch("etna.pipeline.pipeline.Pipeline.load")
def test_load_no_ts(load_mock, product_level_constant_hierarchical_ts, tmp_path):
    ts = product_level_constant_hierarchical_ts
    model = NaiveModel()
    reconciliator = BottomUpReconciliator(target_level="market", source_level="product")
    pipeline = HierarchicalPipeline(reconciliator=reconciliator, model=model, transforms=[], horizon=1)
    dir_path = pathlib.Path(tmp_path)
    path = dir_path / "dummy.zip"
    pipeline.fit(ts)

    pipeline.save(path)
    loaded_pipeline = HierarchicalPipeline.load(path)

    load_mock.assert_called_once_with(path=path)
    assert loaded_pipeline._fit_ts is None
    assert loaded_pipeline.ts is None
    assert loaded_pipeline == load_mock.return_value


@patch("etna.pipeline.pipeline.Pipeline.load")
def test_load_with_ts(load_mock, product_level_constant_hierarchical_ts, tmp_path):
    ts = product_level_constant_hierarchical_ts
    model = NaiveModel()
    reconciliator = BottomUpReconciliator(target_level="market", source_level="product")
    pipeline = HierarchicalPipeline(reconciliator=reconciliator, model=model, transforms=[], horizon=1)
    dir_path = pathlib.Path(tmp_path)
    path = dir_path / "dummy.zip"
    pipeline.fit(ts)

    pipeline.save(path)
    loaded_pipeline = HierarchicalPipeline.load(path, ts=ts)

    load_mock.assert_called_once_with(path=path)
    load_mock.return_value.reconciliator.aggregate.assert_called_once_with(ts=ts)
    pd.testing.assert_frame_equal(loaded_pipeline._fit_ts.to_pandas(), ts.to_pandas())
    assert loaded_pipeline.ts == load_mock.return_value.reconciliator.aggregate.return_value


@pytest.mark.parametrize(
    "reconciliator",
    (
        TopDownReconciliator(target_level="product", source_level="market", period=1, method="AHP"),
        TopDownReconciliator(target_level="product", source_level="market", period=1, method="PHA"),
        BottomUpReconciliator(target_level="market", source_level="product"),
        BottomUpReconciliator(target_level="total", source_level="market"),
    ),
)
@pytest.mark.parametrize(
    "model, transforms",
    [
        (
            CatBoostMultiSegmentModel(iterations=100),
            [DateFlagsTransform(), LagTransform(in_column="target", lags=[1])],
        ),
        (
            LinearPerSegmentModel(),
            [DateFlagsTransform(), LagTransform(in_column="target", lags=[1])],
        ),
        (NaiveModel(), []),
        (ProphetModel(), []),
    ],
)
def test_save_load(model, transforms, reconciliator, product_level_constant_hierarchical_ts):
    horizon = 1
    pipeline = HierarchicalPipeline(reconciliator=reconciliator, model=model, transforms=transforms, horizon=horizon)
    assert_pipeline_equals_loaded_original(pipeline=pipeline, ts=product_level_constant_hierarchical_ts)


@pytest.mark.parametrize(
    "reconciliator",
    (
        TopDownReconciliator(target_level="product", source_level="market", period=1, method="AHP"),
        TopDownReconciliator(target_level="product", source_level="market", period=1, method="PHA"),
        BottomUpReconciliator(target_level="market", source_level="product"),
        BottomUpReconciliator(target_level="total", source_level="market"),
    ),
)
@pytest.mark.parametrize(
    "model, transforms",
    [
        (
            CatBoostMultiSegmentModel(iterations=100),
            [DateFlagsTransform(), LagTransform(in_column="target", lags=[1])],
        ),
        (
            LinearPerSegmentModel(),
            [DateFlagsTransform(), LagTransform(in_column="target", lags=[1])],
        ),
        (NaiveModel(), []),
        (ProphetModel(), []),
    ],
)
def test_forecast_given_ts(model, transforms, reconciliator, product_level_constant_hierarchical_ts):
    horizon = 1
    pipeline = HierarchicalPipeline(reconciliator=reconciliator, model=model, transforms=transforms, horizon=horizon)
    assert_pipeline_forecasts_given_ts(pipeline=pipeline, ts=product_level_constant_hierarchical_ts, horizon=horizon)


@pytest.mark.parametrize(
    "reconciliator",
    (
        TopDownReconciliator(target_level="product", source_level="market", period=1, method="AHP"),
        TopDownReconciliator(target_level="product", source_level="market", period=1, method="PHA"),
        BottomUpReconciliator(target_level="market", source_level="product"),
        BottomUpReconciliator(target_level="total", source_level="market"),
    ),
)
@pytest.mark.parametrize(
    "model, transforms",
    [
        (
            CatBoostMultiSegmentModel(iterations=100),
            [DateFlagsTransform(), LagTransform(in_column="target", lags=[1])],
        ),
        (
            LinearPerSegmentModel(),
            [DateFlagsTransform(), LagTransform(in_column="target", lags=[1])],
        ),
        (NaiveModel(), []),
        (ProphetModel(), []),
    ],
)
def test_forecast_given_ts_with_prediction_interval(
    model, transforms, reconciliator, product_level_constant_hierarchical_ts
):
    horizon = 1
    pipeline = HierarchicalPipeline(reconciliator=reconciliator, model=model, transforms=transforms, horizon=horizon)
    assert_pipeline_forecasts_given_ts_with_prediction_intervals(
        pipeline=pipeline, ts=product_level_constant_hierarchical_ts, horizon=horizon, n_folds=2
    )


@pytest.mark.parametrize(
    "model_fixture",
    (
        "non_prediction_interval_context_ignorant_dummy_model",
        "non_prediction_interval_context_required_dummy_model",
        "prediction_interval_context_ignorant_dummy_model",
        "prediction_interval_context_required_dummy_model",
    ),
)
def test_forecast_with_return_components(
    product_level_constant_hierarchical_ts, model_fixture, request, expected_component_a=20, expected_component_b=180
):
    model = request.getfixturevalue(model_fixture)
    pipeline = HierarchicalPipeline(
        reconciliator=BottomUpReconciliator(target_level="market", source_level="product"), model=model
    )
    pipeline.fit(product_level_constant_hierarchical_ts)
    forecast = pipeline.forecast(return_components=True)

    assert sorted(forecast.target_components_names) == sorted(["target_component_a", "target_component_b"])

    target_components_df = TSDataset.to_flatten(forecast.get_target_components())
    assert (target_components_df["target_component_a"] == expected_component_a).all()
    assert (target_components_df["target_component_b"] == expected_component_b).all()


@pytest.mark.parametrize(
    "model_fixture",
    (
        "non_prediction_interval_context_ignorant_dummy_model",
        "non_prediction_interval_context_required_dummy_model",
        "prediction_interval_context_ignorant_dummy_model",
        "prediction_interval_context_required_dummy_model",
    ),
)
def test_predict_with_return_components(
    product_level_constant_hierarchical_ts, model_fixture, request, expected_component_a=40, expected_component_b=360
):
    ts = product_level_constant_hierarchical_ts
    model = request.getfixturevalue(model_fixture)

    pipeline = HierarchicalPipeline(
        reconciliator=BottomUpReconciliator(target_level="market", source_level="product"), model=model
    )
    pipeline.fit(ts)
    forecast = pipeline.predict(ts=ts, start_timestamp=ts.index[1], return_components=True)

    assert sorted(forecast.target_components_names) == sorted(["target_component_a", "target_component_b"])

    target_components_df = TSDataset.to_flatten(forecast.get_target_components())
    assert (target_components_df["target_component_a"] == expected_component_a).all()
    assert (target_components_df["target_component_b"] == expected_component_b).all()


@pytest.mark.parametrize(
    "reconciliator",
    (
        TopDownReconciliator(target_level="product", source_level="market", period=1, method="AHP"),
        TopDownReconciliator(target_level="product", source_level="market", period=1, method="PHA"),
        BottomUpReconciliator(target_level="market", source_level="product"),
        BottomUpReconciliator(target_level="total", source_level="market"),
    ),
)
@pytest.mark.parametrize(
    "model, transforms, expected_params_to_tune",
    [
        (
            CatBoostMultiSegmentModel(iterations=100),
            [DateFlagsTransform(), LagTransform(in_column="target", lags=list(range(3, 10)))],
            {
                "model.learning_rate": FloatDistribution(low=1e-4, high=0.5, log=True),
                "model.depth": IntDistribution(low=1, high=11, step=1),
                "model.l2_leaf_reg": FloatDistribution(low=0.1, high=200.0, log=True),
                "model.random_strength": FloatDistribution(low=1e-05, high=10.0, log=True),
                "transforms.0.day_number_in_week": CategoricalDistribution([False, True]),
                "transforms.0.day_number_in_month": CategoricalDistribution([False, True]),
                "transforms.0.day_number_in_year": CategoricalDistribution([False, True]),
                "transforms.0.week_number_in_month": CategoricalDistribution([False, True]),
                "transforms.0.week_number_in_year": CategoricalDistribution([False, True]),
                "transforms.0.month_number_in_year": CategoricalDistribution([False, True]),
                "transforms.0.season_number": CategoricalDistribution([False, True]),
                "transforms.0.year_number": CategoricalDistribution([False, True]),
                "transforms.0.is_weekend": CategoricalDistribution([False, True]),
            },
        ),
    ],
)
def test_params_to_tune(reconciliator, model, transforms, expected_params_to_tune):
    horizon = 1
    pipeline = HierarchicalPipeline(reconciliator=reconciliator, model=model, transforms=transforms, horizon=horizon)

    obtained_params_to_tune = pipeline.params_to_tune()

    assert obtained_params_to_tune == expected_params_to_tune
