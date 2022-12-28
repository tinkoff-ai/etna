import numpy as np
import pytest

from etna.datasets import TSDataset
from etna.reconciliation.top_down import TopDownReconciler


@pytest.fixture
def total_level_simple_hierarchical_ts(total_level_df, hierarchical_structure):
    ts = TSDataset(df=total_level_df, freq="D", hierarchical_structure=hierarchical_structure)
    return ts


@pytest.fixture
def market_level_simple_hierarchical_ts(market_level_df, hierarchical_structure):
    ts = TSDataset(df=market_level_df, freq="D", hierarchical_structure=hierarchical_structure)
    return ts


@pytest.fixture
def product_level_simple_hierarchical_ts(product_level_df, hierarchical_structure):
    ts = TSDataset(df=product_level_df, freq="D", hierarchical_structure=hierarchical_structure)
    return ts


@pytest.fixture
def simple_no_hierarchy_ts(market_level_df):
    ts = TSDataset(df=market_level_df, freq="D")
    return ts


@pytest.mark.parametrize(
    "reconciler_args,error_message",
    (
        (
            {
                "period_length": 0,
                "method": "AHP",
            },
            "Period length must be positive!",
        ),
        (
            {
                "period_length": -1,
                "method": "AHP",
            },
            "Period length must be positive!",
        ),
        (
            {
                "period_length": 1,
                "method": "ahp",
            },
            "Unable to recognize reconciliation method 'ahp'! Supported methods: AHP, PHA.",
        ),
        (
            {"period_length": 1, "method": ""},
            "Unable to recognize reconciliation method ''! Supported methods: AHP, PHA.",
        ),
        (
            {
                "period_length": 1,
                "method": "abcd",
            },
            "Unable to recognize reconciliation method 'abcd'! Supported methods: AHP, PHA.",
        ),
    ),
)
def test_top_down_reconcile_init_error(reconciler_args, error_message):
    reconciler_args["target_level"] = "total"
    reconciler_args["source_level"] = "market"

    with pytest.raises(ValueError, match=error_message):
        TopDownReconciler(**reconciler_args)


@pytest.mark.parametrize(
    "ts_name,reconciler_args,error_message",
    (
        (
            "product_level_simple_hierarchical_ts",
            {
                "target_level": "total",
                "source_level": "market",
            },
            "Target level should be lower or equal in the hierarchy than the source level!",
        ),
        (
            "market_level_simple_hierarchical_ts",
            {
                "target_level": "product",
                "source_level": "total",
            },
            "Current TSDataset level should be lower or equal in the hierarchy than the target level!",
        ),
        (
            "total_level_simple_hierarchical_ts",
            {
                "target_level": "product",
                "source_level": "market",
            },
            "Current TSDataset level should be lower or equal in the hierarchy than the target level!",
        ),
        (
            "market_level_simple_hierarchical_ts",
            {
                "target_level": "abc",
                "source_level": "total",
            },
            "Invalid level name: abc",
        ),
        (
            "market_level_simple_hierarchical_ts",
            {
                "target_level": "market",
                "source_level": "abc",
            },
            "Invalid level name: abc",
        ),
        (
            "simple_no_hierarchy_ts",
            {
                "target_level": "total",
                "source_level": "market",
            },
            "The method can be applied only to instances with a hierarchy!",
        ),
    ),
)
def test_top_down_reconcile_errors(ts_name, reconciler_args, error_message, request):
    ts = request.getfixturevalue(ts_name)
    reconciler_args["period_length"] = 1
    reconciler_args["method"] = "AHP"

    reconciler = TopDownReconciler(**reconciler_args)
    with pytest.raises(ValueError, match=error_message):
        reconciler.fit(ts)


@pytest.mark.parametrize(
    "ts_name,reconciler_args,answer",
    (
        (
            "product_level_simple_hierarchical_ts",
            {
                "method": "AHP",
                "period_length": 1,
                "target_level": "product",
                "source_level": "product",
            },
            np.identity(4),
        ),
        (
            "product_level_simple_hierarchical_ts",
            {
                "method": "AHP",
                "period_length": 1,
                "target_level": "product",
                "source_level": "market",
            },
            np.array([[0.5, 0], [0.5, 0], [0, 0.9], [0, 0.1]]),
        ),
        (
            "product_level_simple_hierarchical_ts",
            {
                "method": "AHP",
                "period_length": 1,
                "target_level": "product",
                "source_level": "total",
            },
            np.array([[0.04545], [0.04545], [0.8182], [0.0909]]),
        ),
        (
            "product_level_simple_hierarchical_ts",
            {
                "method": "AHP",
                "period_length": 1,
                "target_level": "market",
                "source_level": "total",
            },
            np.array([[0.0909], [0.9091]]),
        ),
        (
            "market_level_simple_hierarchical_ts",
            {
                "method": "AHP",
                "period_length": 1,
                "target_level": "market",
                "source_level": "market",
            },
            np.array([[1, 0.0], [0.0, 1]]),
        ),
        (
            "market_level_simple_hierarchical_ts",
            {
                "method": "AHP",
                "period_length": 1,
                "target_level": "market",
                "source_level": "total",
            },
            np.array([[0.0909], [0.9091]]),
        ),
        (
            "total_level_simple_hierarchical_ts",
            {
                "method": "AHP",
                "period_length": 1,
                "target_level": "total",
                "source_level": "total",
            },
            np.array([[1]]),
        ),
        (
            "product_level_simple_hierarchical_ts",
            {
                "method": "PHA",
                "period_length": 1,
                "target_level": "product",
                "source_level": "product",
            },
            np.identity(4),
        ),
        (
            "product_level_simple_hierarchical_ts",
            {
                "method": "PHA",
                "period_length": 1,
                "target_level": "product",
                "source_level": "market",
            },
            np.array([[0.5, 0], [0.5, 0], [0, 0.9], [0, 0.1]]),
        ),
        (
            "product_level_simple_hierarchical_ts",
            {
                "method": "PHA",
                "period_length": 1,
                "target_level": "product",
                "source_level": "total",
            },
            np.array([[0.04545], [0.04545], [0.8182], [0.0909]]),
        ),
        (
            "product_level_simple_hierarchical_ts",
            {
                "method": "PHA",
                "period_length": 1,
                "target_level": "market",
                "source_level": "total",
            },
            np.array([[0.0909], [0.9091]]),
        ),
        (
            "market_level_simple_hierarchical_ts",
            {
                "method": "PHA",
                "period_length": 1,
                "target_level": "market",
                "source_level": "market",
            },
            np.array([[1, 0.0], [0.0, 1]]),
        ),
        (
            "market_level_simple_hierarchical_ts",
            {
                "method": "PHA",
                "period_length": 1,
                "target_level": "market",
                "source_level": "total",
            },
            np.array([[0.0909], [0.9091]]),
        ),
        (
            "total_level_simple_hierarchical_ts",
            {
                "method": "PHA",
                "period_length": 1,
                "target_level": "total",
                "source_level": "total",
            },
            np.array([[1]]),
        ),
        (
            "product_level_simple_hierarchical_ts",
            {
                "method": "AHP",
                "period_length": 2,
                "target_level": "product",
                "source_level": "product",
            },
            np.identity(4),
        ),
        (
            "product_level_simple_hierarchical_ts",
            {
                "method": "AHP",
                "period_length": 2,
                "target_level": "product",
                "source_level": "market",
            },
            np.array([[0.75, 0], [0.25, 0], [0, 0.6], [0, 0.4]]),
        ),
        (
            "product_level_simple_hierarchical_ts",
            {
                "method": "AHP",
                "period_length": 2,
                "target_level": "product",
                "source_level": "total",
            },
            np.array([[0.0682], [0.0227], [0.5455], [0.3636]]),
        ),
        (
            "product_level_simple_hierarchical_ts",
            {
                "method": "AHP",
                "period_length": 2,
                "target_level": "market",
                "source_level": "total",
            },
            np.array([[0.0909], [0.9091]]),
        ),
        (
            "market_level_simple_hierarchical_ts",
            {
                "method": "AHP",
                "period_length": 2,
                "target_level": "market",
                "source_level": "market",
            },
            np.identity(2),
        ),
        (
            "market_level_simple_hierarchical_ts",
            {
                "method": "AHP",
                "period_length": 2,
                "target_level": "market",
                "source_level": "total",
            },
            np.array([[0.0909], [0.9091]]),
        ),
        (
            "total_level_simple_hierarchical_ts",
            {
                "method": "AHP",
                "period_length": 2,
                "target_level": "total",
                "source_level": "total",
            },
            np.array([[1]]),
        ),
        (
            "product_level_simple_hierarchical_ts",
            {
                "method": "PHA",
                "period_length": 2,
                "target_level": "product",
                "source_level": "product",
            },
            np.identity(4),
        ),
        (
            "product_level_simple_hierarchical_ts",
            {
                "method": "PHA",
                "period_length": 2,
                "target_level": "product",
                "source_level": "market",
            },
            np.array([[0.6667, 0], [0.3333, 0], [0, 0.7], [0, 0.3]]),
        ),
        (
            "product_level_simple_hierarchical_ts",
            {
                "method": "PHA",
                "period_length": 2,
                "target_level": "product",
                "source_level": "total",
            },
            np.array([[0.0606], [0.0303], [0.6364], [0.2727]]),
        ),
        (
            "product_level_simple_hierarchical_ts",
            {
                "method": "PHA",
                "period_length": 2,
                "target_level": "market",
                "source_level": "total",
            },
            np.array([[0.0909], [0.9091]]),
        ),
        (
            "market_level_simple_hierarchical_ts",
            {
                "method": "PHA",
                "period_length": 2,
                "target_level": "market",
                "source_level": "market",
            },
            np.identity(2),
        ),
        (
            "market_level_simple_hierarchical_ts",
            {
                "method": "PHA",
                "period_length": 2,
                "target_level": "market",
                "source_level": "total",
            },
            np.array([[0.0909], [0.9091]]),
        ),
        (
            "total_level_simple_hierarchical_ts",
            {
                "method": "PHA",
                "period_length": 2,
                "target_level": "total",
                "source_level": "total",
            },
            np.array([[1]]),
        ),
    ),
)
def test_top_down_reconcile_fit(ts_name, reconciler_args, answer, request):
    ts = request.getfixturevalue(ts_name)
    reconciler = TopDownReconciler(**reconciler_args)
    reconciler.fit(ts)
    np.testing.assert_array_almost_equal(reconciler.mapping_matrix.toarray().round(5), answer, decimal=4)
