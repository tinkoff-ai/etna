import numpy as np
import pandas as pd
import pytest
from sklearn.neighbors import KNeighborsClassifier
from tsfresh.feature_extraction.settings import MinimalFCParameters

from etna.datasets import TSDataset
from etna.datasets import generate_ar_df
from etna.experimental.classification.feature_extraction.tsfresh import TSFreshFeatureExtractor
from etna.experimental.classification.predictability import PredictabilityAnalyzer


@pytest.fixture
def many_time_series_ts(many_time_series):
    x, y = many_time_series
    dfs = []
    ts_y = {}
    for i, series in enumerate(x):
        df = generate_ar_df(periods=10, n_segments=1, start_time="2000-01-01")
        df = df.iloc[-len(series) :]
        df["target"] = series
        df["segment"] = f"segment_{i}"
        ts_y[f"segment_{i}"] = y[i]
        dfs.append(df)
    df = pd.concat(dfs)
    df = TSDataset.to_dataset(df)
    ts = TSDataset(df=df, freq="D")
    return ts, ts_y


def test_get_series_from_dataset(many_time_series, many_time_series_ts):
    ts, _ = many_time_series_ts
    x = PredictabilityAnalyzer.get_series_from_dataset(ts=ts)
    x_expected, _ = many_time_series
    for row, row_expected in zip(x, x_expected):
        np.testing.assert_array_equal(row, row_expected)


def test_analyze(many_time_series, many_time_series_ts):
    x, y = many_time_series
    analyzer = PredictabilityAnalyzer(
        feature_extractor=TSFreshFeatureExtractor(default_fc_parameters=MinimalFCParameters()),
        classifier=KNeighborsClassifier(n_neighbors=1),
    )
    analyzer.fit(x=x, y=y)

    ts, ts_y = many_time_series_ts
    result = analyzer.analyze_predictability(ts=ts)
    assert isinstance(result, dict)
    assert sorted(result.keys()) == sorted(ts.segments)
    for segment in ts.segments:
        assert result[segment] == ts_y[segment]
