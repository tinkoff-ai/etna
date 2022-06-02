import numpy as np
import pytest

from etna.analysis import get_anomalies_prediction_interval
from etna.analysis.outliers.prediction_interval_outliers import create_ts_by_column
from etna.datasets import TSDataset
from etna.models import ProphetModel
from etna.models import SARIMAXModel


@pytest.mark.parametrize("column", ["exog"])
def test_create_ts_by_column_interface(outliers_tsds, column):
    """Test that `create_ts_column` produces correct columns."""
    new_ts = create_ts_by_column(outliers_tsds, column)
    assert isinstance(new_ts, TSDataset)
    assert outliers_tsds.segments == new_ts.segments
    assert new_ts.columns.get_level_values("feature").unique().tolist() == ["target"]


@pytest.mark.parametrize("column", ["exog"])
def test_create_ts_by_column_retain_column(outliers_tsds, column):
    """Test that `create_ts_column` selects correct data in selected columns."""
    new_ts = create_ts_by_column(outliers_tsds, column)
    for segment in new_ts.segments:
        new_series = new_ts[:, segment, "target"]
        original_series = outliers_tsds[:, segment, column]
        new_series = new_series[~new_series.isna()]
        original_series = original_series[~original_series.isna()]
        assert np.all(new_series == original_series)


@pytest.mark.parametrize("in_column", ["target", "exog"])
@pytest.mark.parametrize("model", (ProphetModel, SARIMAXModel))
def test_get_anomalies_prediction_interval_interface(outliers_tsds, model, in_column):
    """Test that `get_anomalies_prediction_interval` produces correct columns."""
    anomalies = get_anomalies_prediction_interval(outliers_tsds, model=model, interval_width=0.95, in_column=in_column)
    assert isinstance(anomalies, dict)
    assert sorted(anomalies.keys()) == sorted(outliers_tsds.segments)
    for segment in anomalies.keys():
        assert isinstance(anomalies[segment], list)
        for date in anomalies[segment]:
            assert isinstance(date, np.datetime64)


@pytest.mark.parametrize("in_column", ["target", "exog"])
@pytest.mark.parametrize(
    "model, interval_width, true_anomalies",
    (
        (
            ProphetModel,
            0.95,
            {"1": [np.datetime64("2021-01-11")], "2": [np.datetime64("2021-01-09"), np.datetime64("2021-01-27")]},
        ),
        (SARIMAXModel, 0.999, {"1": [], "2": [np.datetime64("2021-01-27")]}),
    ),
)
def test_get_anomalies_prediction_interval_values(outliers_tsds, model, interval_width, true_anomalies, in_column):
    """Test that `get_anomalies_prediction_interval` generates correct values."""
    assert (
        get_anomalies_prediction_interval(
            outliers_tsds, model=model, interval_width=interval_width, in_column=in_column
        )
        == true_anomalies
    )
