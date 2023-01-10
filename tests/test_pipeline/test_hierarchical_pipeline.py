from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from etna.datasets import TSDataset
from etna.metrics import MAE
from etna.models import LinearPerSegmentModel
from etna.models import NaiveModel
from etna.pipeline.hierarchical_pipeline import HierarchicalPipeline
from etna.reconciliation import BottomUpReconciliator
from etna.reconciliation import TopDownReconciliator
from etna.transforms import LinearTrendTransform
from etna.transforms import MeanTransform


@pytest.fixture
def product_level_constant_hierarchical_df():
    df = pd.DataFrame(
        {
            "timestamp": ["2000-01-01", "2000-01-02", "2000-01-03", "2000-01-04"] * 4,
            "segment": ["a"] * 4 + ["b"] * 4 + ["c"] * 4 + ["d"] * 4,
            "target": [1, 1, 1, 1] + [2, 2, 2, 2] + [3, 3, 3, 3] + [4, 4, 4, 4],
        }
    )
    df = TSDataset.to_dataset(df)
    return df


@pytest.fixture
def market_level_constant_hierarchical_df():
    df = pd.DataFrame(
        {
            "timestamp": ["2000-01-01", "2000-01-02", "2000-01-03", "2000-01-04"] * 2,
            "segment": ["X"] * 4 + ["Y"] * 4,
            "target": [3, 3, 3, 3] + [7, 7, 7, 7],
        }
    )
    df = TSDataset.to_dataset(df)
    return df


@pytest.fixture
def market_level_constant_hierarchical_df_exog():
    df = pd.DataFrame(
        {
            "timestamp": ["2000-01-01", "2000-01-02", "2000-01-03", "2000-01-04", "2000-01-05", "2000-01-06"] * 2,
            "segment": ["X"] * 6 + ["Y"] * 6,
            "regressor": [1, 1, 1, 1, 1, 1] * 2,
        }
    )
    df = TSDataset.to_dataset(df)
    return df


@pytest.fixture
def market_level_constant_hierarchical_ts(market_level_constant_hierarchical_df, hierarchical_structure):
    ts = TSDataset(df=market_level_constant_hierarchical_df, freq="D", hierarchical_structure=hierarchical_structure)
    return ts


@pytest.fixture
def market_level_constant_hierarchical_ts_w_exog(
    market_level_constant_hierarchical_df, market_level_constant_hierarchical_df_exog, hierarchical_structure
):
    ts = TSDataset(
        df=market_level_constant_hierarchical_df,
        df_exog=market_level_constant_hierarchical_df_exog,
        freq="D",
        hierarchical_structure=hierarchical_structure,
        known_future="all",
    )
    return ts


@pytest.fixture
def product_level_constant_hierarchical_ts_w_exog(
    product_level_constant_hierarchical_df, market_level_constant_hierarchical_df_exog, hierarchical_structure
):
    ts = TSDataset(
        df=product_level_constant_hierarchical_df,
        df_exog=market_level_constant_hierarchical_df_exog,
        freq="D",
        hierarchical_structure=hierarchical_structure,
        known_future="all",
    )
    return ts


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
    with patch.object(type(reconciliator), "fit", wraps=reconciliator.fit) as mock:
        pipeline.fit(market_level_simple_hierarchical_ts)
        mock.assert_called()


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
    forecast = pipeline.raw_forecast()
    np.testing.assert_array_almost_equal(forecast[..., "target"].values, answer)


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
    forecast = pipeline.raw_forecast()
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
    np.testing.assert_array_almost_equal(forecast[..., "target"].values, answer)


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
        TopDownReconciliator(target_level="market", source_level="total", period=1, method="PHA"),
        BottomUpReconciliator(target_level="total", source_level="market"),
    ),
)
def test_backtest(market_level_constant_hierarchical_ts, reconciliator):
    ts = market_level_constant_hierarchical_ts
    model = NaiveModel()
    pipeline = HierarchicalPipeline(reconciliator=reconciliator, model=model, transforms=[], horizon=1)
    metrics, _, _ = pipeline.backtest(ts=ts, metrics=[MAE()], n_folds=2, aggregate_metrics=True)
    np.testing.assert_array_almost_equal(metrics["MAE"], 0)


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
    model = NaiveModel()
    transforms = [MeanTransform(in_column="target", window=2), LinearTrendTransform(in_column="target")]
    pipeline = HierarchicalPipeline(reconciliator=reconciliator, model=model, transforms=transforms, horizon=1)
    metrics, _, _ = pipeline.backtest(ts=ts, metrics=[MAE()], n_folds=2, aggregate_metrics=True)
    np.testing.assert_array_almost_equal(metrics["MAE"], 0)


@pytest.mark.parametrize(
    "reconciliator",
    (
        TopDownReconciliator(target_level="market", source_level="total", period=1, method="AHP"),
        TopDownReconciliator(target_level="market", source_level="total", period=1, method="PHA"),
        BottomUpReconciliator(target_level="total", source_level="market"),
    ),
)
def test_backtest_w_exog(market_level_constant_hierarchical_ts_w_exog, reconciliator):
    ts = market_level_constant_hierarchical_ts_w_exog
    model = NaiveModel()
    pipeline = HierarchicalPipeline(reconciliator=reconciliator, model=model, transforms=[], horizon=1)
    metrics, _, _ = pipeline.backtest(ts=ts, metrics=[MAE()], n_folds=2, aggregate_metrics=True)
    np.testing.assert_array_almost_equal(metrics["MAE"], 0)


@pytest.mark.parametrize(
    "reconciliator",
    (
        TopDownReconciliator(target_level="product", source_level="market", period=1, method="AHP"),
        TopDownReconciliator(target_level="product", source_level="market", period=1, method="PHA"),
        BottomUpReconciliator(target_level="total", source_level="market"),
    ),
)
def test_backtest_w_regressors(product_level_constant_hierarchical_ts_w_exog, reconciliator):
    ts = product_level_constant_hierarchical_ts_w_exog
    model = LinearPerSegmentModel()
    pipeline = HierarchicalPipeline(reconciliator=reconciliator, model=model, transforms=[], horizon=1)
    metrics, _, _ = pipeline.backtest(ts=ts, metrics=[MAE()], n_folds=2, aggregate_metrics=True)
    np.testing.assert_array_almost_equal(metrics["MAE"], 0)
