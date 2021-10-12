import numpy as np
import pytest

from etna.analysis import get_anomalies_confidence_interval
from etna.models import ProphetModel
from etna.models import SARIMAXModel


@pytest.mark.parametrize("model", (ProphetModel, SARIMAXModel))
def test_interface(outliers_tsds, model):
    anomalies = get_anomalies_confidence_interval(outliers_tsds, model=model, interval_width=0.95)
    assert isinstance(anomalies, dict)
    assert sorted(list(anomalies.keys())) == sorted(outliers_tsds.segments)
    for segment in anomalies.keys():
        assert isinstance(anomalies[segment], list)
        for date in anomalies[segment]:
            assert isinstance(date, np.datetime64)


@pytest.mark.parametrize("model", (ProphetModel, SARIMAXModel))
@pytest.mark.parametrize(
    "interval_width, true_anomalies",
    ((0.95, {"1": [np.datetime64("2021-01-11")], "2": [np.datetime64("2021-01-09"), np.datetime64("2021-01-27")]}),),
)
def test_confidence_interval_outliers(outliers_tsds, model, interval_width, true_anomalies):
    assert get_anomalies_confidence_interval(outliers_tsds, model, interval_width) == true_anomalies
