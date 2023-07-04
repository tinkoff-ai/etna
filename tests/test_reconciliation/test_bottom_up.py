import numpy as np
import pandas as pd
import pytest

from etna.reconciliation.bottom_up import BottomUpReconciliator


@pytest.mark.parametrize(
    "target_level,source_level,error_message",
    (
        (
            "market",
            "total",
            "Source level should be lower or equal in the hierarchy than the target level!",
        ),
        (
            "total",
            "product",
            "Current TSDataset level should be lower or equal in the hierarchy than the source level!",
        ),
    ),
)
def test_bottom_up_reconcile_level_order_errors(
    market_level_simple_hierarchical_ts, target_level, source_level, error_message
):
    reconciler = BottomUpReconciliator(target_level=target_level, source_level=source_level)
    with pytest.raises(ValueError, match=error_message):
        reconciler.fit(market_level_simple_hierarchical_ts)


@pytest.mark.parametrize(
    "target_level,source_level",
    (("abc", "total"), ("market", "abc")),
)
def test_bottom_up_reconcile_invalid_level_errors(market_level_simple_hierarchical_ts, target_level, source_level):
    reconciler = BottomUpReconciliator(target_level=target_level, source_level=source_level)
    with pytest.raises(ValueError, match="Invalid level name: abc"):
        reconciler.fit(market_level_simple_hierarchical_ts)


def test_bottom_up_reconcile_no_hierarchy_error(simple_no_hierarchy_ts):
    reconciler = BottomUpReconciliator(target_level="market", source_level="total")
    with pytest.raises(ValueError, match="The method can be applied only to instances with a hierarchy!"):
        reconciler.fit(simple_no_hierarchy_ts)


@pytest.mark.parametrize(
    "ts_name,target_level,source_level,answer",
    (
        (
            "product_level_simple_hierarchical_ts",
            "product",
            "product",
            np.identity(4),
        ),
        (
            "product_level_simple_hierarchical_ts",
            "market",
            "product",
            np.array([[1, 1, 0, 0], [0, 0, 1, 1]]),
        ),
        (
            "product_level_simple_hierarchical_ts",
            "total",
            "product",
            np.array([[1, 1, 1, 1]]),
        ),
        (
            "product_level_simple_hierarchical_ts",
            "total",
            "market",
            np.array([[1, 1]]),
        ),
        (
            "market_level_simple_hierarchical_ts",
            "total",
            "market",
            np.array([[1, 1]]),
        ),
        (
            "total_level_simple_hierarchical_ts",
            "total",
            "total",
            np.array([[1]]),
        ),
    ),
)
def test_bottom_up_reconcile_fit(ts_name, target_level, source_level, answer, request):
    ts = request.getfixturevalue(ts_name)
    reconciler = BottomUpReconciliator(target_level=target_level, source_level=source_level)
    reconciler.fit(ts)
    np.testing.assert_array_almost_equal(reconciler.mapping_matrix.toarray().round(5), answer, decimal=4)


def test_bottom_up_reconcile_fit_w_nans(market_level_simple_hierarchical_ts_w_nans):
    answer = np.array([[1, 1]])
    reconciler = BottomUpReconciliator(source_level="market", target_level="total")
    reconciler.fit(market_level_simple_hierarchical_ts_w_nans)
    np.testing.assert_array_almost_equal(reconciler.mapping_matrix.toarray().round(5), answer, decimal=4)


def test_reconcile_with_quantiles(
    product_level_constant_forecast_with_quantiles,
    market_level_constant_forecast_with_quantiles,
    product_level_constant_hierarchical_ts,
):
    ts = product_level_constant_hierarchical_ts
    reconciliator = BottomUpReconciliator(target_level="market", source_level="product")
    reconciliator.fit(ts=ts)
    reconciled_df = reconciliator.reconcile(ts=product_level_constant_forecast_with_quantiles).to_pandas()
    pd.testing.assert_frame_equal(reconciled_df, market_level_constant_forecast_with_quantiles.to_pandas())
