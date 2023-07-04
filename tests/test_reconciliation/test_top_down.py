import numpy as np
import pytest

from etna.reconciliation.top_down import TopDownReconciliator


@pytest.mark.parametrize(
    "period",
    (0, -1),
)
def test_top_down_reconcile_init_period_error(period):
    with pytest.raises(ValueError, match="Period length must be positive!"):
        TopDownReconciliator(period=period, method="market", target_level="market", source_level="total")


@pytest.mark.parametrize(
    "method",
    ("abcd", ""),
)
def test_top_down_reconcile_init_method_error(method):
    with pytest.raises(
        ValueError, match=f"Unable to recognize reconciliation method '{method}'! Supported methods: AHP, PHA."
    ):
        TopDownReconciliator(period=1, method=method, target_level="market", source_level="total")


@pytest.mark.parametrize(
    "ts_name,target_level,source_level,error_message",
    (
        (
            "product_level_simple_hierarchical_ts",
            "total",
            "market",
            "Target level should be lower or equal in the hierarchy than the source level!",
        ),
        (
            "market_level_simple_hierarchical_ts",
            "product",
            "total",
            "Current TSDataset level should be lower or equal in the hierarchy than the target level!",
        ),
    ),
)
def test_top_down_reconcile_level_order_errors(ts_name, target_level, source_level, error_message, request):
    ts = request.getfixturevalue(ts_name)
    reconciler = TopDownReconciliator(period=1, method="AHP", target_level=target_level, source_level=source_level)
    with pytest.raises(ValueError, match=error_message):
        reconciler.fit(ts)


@pytest.mark.parametrize(
    "target_level,source_level",
    (("abc", "total"), ("market", "abc")),
)
def test_top_down_reconcile_invalid_level_errors(market_level_simple_hierarchical_ts, target_level, source_level):
    reconciler = TopDownReconciliator(period=1, method="AHP", target_level=target_level, source_level=source_level)
    with pytest.raises(ValueError, match="Invalid level name: abc"):
        reconciler.fit(market_level_simple_hierarchical_ts)


def test_top_down_reconcile_no_hierarchy_error(simple_no_hierarchy_ts):
    reconciler = TopDownReconciliator(method="AHP", period=1, target_level="market", source_level="total")
    with pytest.raises(ValueError, match="The method can be applied only to instances with a hierarchy!"):
        reconciler.fit(simple_no_hierarchy_ts)


def test_top_down_reconcile_negatives_error(simple_hierarchical_ts_w_negatives):
    reconciler = TopDownReconciliator(method="AHP", period=1, target_level="market", source_level="total")
    with pytest.raises(ValueError, match="Provided dataset should not contain any negative numbers!"):
        reconciler.fit(simple_hierarchical_ts_w_negatives)


@pytest.mark.parametrize(
    "ts_name,reconciler_args,answer",
    (
        (
            "product_level_simple_hierarchical_ts",
            {
                "period": 1,
                "target_level": "product",
                "source_level": "product",
            },
            np.identity(4),
        ),
        (
            "product_level_simple_hierarchical_ts",
            {
                "period": 1,
                "target_level": "product",
                "source_level": "market",
            },
            np.array([[0.5, 0], [0.5, 0], [0, 0.9], [0, 0.1]]),
        ),
        (
            "product_level_simple_hierarchical_ts",
            {
                "period": 1,
                "target_level": "product",
                "source_level": "total",
            },
            np.array([[0.04545], [0.04545], [0.8182], [0.0909]]),
        ),
        (
            "product_level_simple_hierarchical_ts",
            {
                "period": 1,
                "target_level": "market",
                "source_level": "total",
            },
            np.array([[0.0909], [0.9091]]),
        ),
        (
            "total_level_simple_hierarchical_ts",
            {
                "period": 1,
                "target_level": "total",
                "source_level": "total",
            },
            np.array([[1]]),
        ),
        (
            "product_level_simple_hierarchical_ts",
            {
                "period": 2,
                "target_level": "product",
                "source_level": "market",
            },
            np.array([[0.75, 0], [0.25, 0], [0, 0.6], [0, 0.4]]),
        ),
        (
            "product_level_simple_hierarchical_ts",
            {
                "period": 2,
                "target_level": "product",
                "source_level": "total",
            },
            np.array([[0.0682], [0.0227], [0.5455], [0.3636]]),
        ),
        (
            "product_level_simple_hierarchical_ts",
            {
                "period": 2,
                "target_level": "market",
                "source_level": "total",
            },
            np.array([[0.0909], [0.9091]]),
        ),
    ),
)
def test_top_down_reconcile_ahp_fit(ts_name, reconciler_args, answer, request):
    reconciler_args["method"] = "AHP"
    ts = request.getfixturevalue(ts_name)
    reconciler = TopDownReconciliator(**reconciler_args)
    reconciler.fit(ts)
    np.testing.assert_array_almost_equal(reconciler.mapping_matrix.toarray().round(5), answer, decimal=4)


@pytest.mark.parametrize(
    "ts_name,reconciler_args,answer",
    (
        (
            "product_level_simple_hierarchical_ts",
            {
                "period": 1,
                "target_level": "product",
                "source_level": "product",
            },
            np.identity(4),
        ),
        (
            "product_level_simple_hierarchical_ts",
            {
                "period": 1,
                "target_level": "product",
                "source_level": "market",
            },
            np.array([[0.5, 0], [0.5, 0], [0, 0.9], [0, 0.1]]),
        ),
        (
            "product_level_simple_hierarchical_ts",
            {
                "period": 1,
                "target_level": "product",
                "source_level": "total",
            },
            np.array([[0.04545], [0.04545], [0.8182], [0.0909]]),
        ),
        (
            "product_level_simple_hierarchical_ts",
            {
                "period": 1,
                "target_level": "market",
                "source_level": "total",
            },
            np.array([[0.0909], [0.9091]]),
        ),
        (
            "total_level_simple_hierarchical_ts",
            {
                "period": 1,
                "target_level": "total",
                "source_level": "total",
            },
            np.array([[1]]),
        ),
        (
            "product_level_simple_hierarchical_ts",
            {
                "period": 2,
                "target_level": "product",
                "source_level": "market",
            },
            np.array([[0.6667, 0], [0.3333, 0], [0, 0.7], [0, 0.3]]),
        ),
        (
            "product_level_simple_hierarchical_ts",
            {
                "period": 2,
                "target_level": "product",
                "source_level": "total",
            },
            np.array([[0.0606], [0.0303], [0.6364], [0.2727]]),
        ),
        (
            "product_level_simple_hierarchical_ts",
            {
                "period": 2,
                "target_level": "market",
                "source_level": "total",
            },
            np.array([[0.0909], [0.9091]]),
        ),
    ),
)
def test_top_down_reconcile_pha_fit(ts_name, reconciler_args, answer, request):
    reconciler_args["method"] = "PHA"
    ts = request.getfixturevalue(ts_name)
    reconciler = TopDownReconciliator(**reconciler_args)
    reconciler.fit(ts)
    np.testing.assert_array_almost_equal(reconciler.mapping_matrix.toarray().round(5), answer, decimal=4)


@pytest.mark.parametrize(
    "method,period,answer",
    (
        (
            "AHP",
            1,
            np.array([[np.nan], [np.nan]]),
        ),
        (
            "AHP",
            2,
            np.array([[np.nan], [np.nan]]),
        ),
        (
            "AHP",
            3,
            np.array([[0.1739], [0.8261]]),
        ),
        (
            "AHP",
            4,
            np.array([[0.1739], [0.8261]]),
        ),
        (
            "PHA",
            1,
            np.array([[np.nan], [np.nan]]),
        ),
        (
            "PHA",
            2,
            np.array([[np.nan], [np.nan]]),
        ),
        (
            "PHA",
            3,
            np.array([[0.2174], [0.8913]]),
        ),
        (
            "PHA",
            4,
            np.array([[0.2174], [0.8406]]),
        ),
    ),
)
def test_top_down_reconcile_fit_w_nans(market_level_simple_hierarchical_ts_w_nans, method, period, answer):
    reconciler = TopDownReconciliator(method=method, period=period, source_level="total", target_level="market")
    reconciler.fit(market_level_simple_hierarchical_ts_w_nans)
    np.testing.assert_array_almost_equal(reconciler.mapping_matrix.toarray().round(5), answer, decimal=4)


def test_reconcile_with_quantiles(total_level_constant_forecast_with_quantiles, product_level_constant_hierarchical_ts):
    answer = np.array(
        [
            [10 * 0.3, 6.5 * 0.3, 14 * 0.3, 10 * 0.7, 6.5 * 0.7, 14 * 0.7],
            [10 * 0.3, 3.75 * 0.3, 18 * 0.3, 10 * 0.7, 3.75 * 0.7, 18 * 0.7],
        ]
    )
    ts = product_level_constant_hierarchical_ts
    forecast = total_level_constant_forecast_with_quantiles
    reconciliator = TopDownReconciliator(target_level="market", source_level="total", method="AHP", period=1)
    reconciliator.fit(ts=ts)
    np.testing.assert_array_almost_equal(reconciliator.reconcile(ts=forecast).df.values, answer)
