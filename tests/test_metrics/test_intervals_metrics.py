import pytest

from etna.datasets import TSDataset
from etna.metrics import Coverage
from etna.metrics import Width


@pytest.fixture
def tsdataset_with_zero_width_quantiles(example_df):

    ts_train = TSDataset.to_dataset(example_df)
    ts_train = TSDataset(ts_train, freq="H")
    example_df["target_0.025"] = example_df["target"]
    example_df["target_0.975"] = example_df["target"]
    ts_test = TSDataset.to_dataset(example_df)
    ts_test = TSDataset(ts_test, freq="H")
    return ts_train, ts_test


@pytest.fixture
def tsdataset_with_differnt_width_and_shifted_quantiles(example_df):

    ts_train = TSDataset.to_dataset(example_df)
    ts_train = TSDataset(ts_train, freq="H")
    example_df["target_0.025"] = example_df["target"]
    example_df["target_0.975"] = example_df["target"]

    segment_one_index = example_df[lambda x: x.segment == "segment_1"].index

    example_df.loc[segment_one_index, "target_0.025"] = example_df.loc[segment_one_index, "target_0.025"] + 1
    example_df.loc[segment_one_index, "target_0.975"] = example_df.loc[segment_one_index, "target_0.975"] + 2

    ts_test = TSDataset.to_dataset(example_df)
    ts_test = TSDataset(ts_test, freq="H")
    return ts_train, ts_test


def test_width_metric_with_zero_width_quantiles(tsdataset_with_zero_width_quantiles):
    ts_train, ts_test = tsdataset_with_zero_width_quantiles

    expected_metric = 0.0
    width_metric = Width(mode="per-segment")(ts_train, ts_test)

    for segment in width_metric:
        assert width_metric[segment] == expected_metric


def test_width_metric_with_differnt_width_and_shifted_quantiles(tsdataset_with_differnt_width_and_shifted_quantiles):
    ts_train, ts_test = tsdataset_with_differnt_width_and_shifted_quantiles

    expected_metric = {"segment_1": 1.0, "segment_2": 0.0}
    width_metric = Width(mode="per-segment")(ts_train, ts_test)

    for segment in width_metric:
        assert width_metric[segment] == expected_metric[segment]


def test_coverage_metric_with_differnt_width_and_shifted_quantiles(tsdataset_with_differnt_width_and_shifted_quantiles):
    ts_train, ts_test = tsdataset_with_differnt_width_and_shifted_quantiles

    expected_metric = {"segment_1": 0.0, "segment_2": 1.0}
    coverage_metric = Coverage(mode="per-segment")(ts_train, ts_test)

    for segment in coverage_metric:
        assert coverage_metric[segment] == expected_metric[segment]


@pytest.mark.parametrize("metric", [Coverage(quantiles=(0.1, 0.3)), Width(quantiles=(0.1, 0.3))])
def test_using_not_presented_quantiles(metric, tsdataset_with_zero_width_quantiles):
    ts_train, ts_test = tsdataset_with_zero_width_quantiles
    with pytest.raises(AssertionError, match="Quantile .* is not presented in tsdataset."):
        _ = metric(ts_train, ts_test)


@pytest.mark.parametrize(
    "metric, greater_is_better", ((Coverage(quantiles=(0.1, 0.3)), None), (Width(quantiles=(0.1, 0.3)), False))
)
def test_metrics_greater_is_better(metric, greater_is_better):
    assert metric.greater_is_better == greater_is_better
