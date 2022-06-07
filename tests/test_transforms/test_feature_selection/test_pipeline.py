import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeRegressor

from etna.analysis import StatisticsRelevanceTable
from etna.datasets import TSDataset
from etna.metrics import MAE
from etna.models import ProphetModel
from etna.pipeline import Pipeline
from etna.transforms import LogTransform
from etna.transforms.feature_selection import FilterFeaturesTransform


@pytest.fixture()
def sinusoid_ts():
    periods = 1000
    sinusoid_ts_1 = pd.DataFrame(
        {
            "segment": np.zeros(periods),
            "timestamp": pd.date_range(start="1/1/2018", periods=periods),
            "target": [np.sin(i / 10) + i / 5 for i in range(periods)],
            "feature_1": [i / 10 for i in range(periods)],
            "feature_2": [np.sin(i) for i in range(periods)],
            "feature_3": [np.cos(i / 10) for i in range(periods)],
            "feature_4": [np.cos(i) for i in range(periods)],
            "feature_5": [i ** 2 for i in range(periods)],
            "feature_6": [i * np.sin(i) for i in range(periods)],
            "feature_7": [i * np.cos(i) for i in range(periods)],
            "feature_8": [i + np.cos(i) for i in range(periods)],
        }
    )
    df = TSDataset.to_dataset(sinusoid_ts_1)
    ts = TSDataset(df, freq="D")
    return ts


@pytest.mark.parametrize("relevance_table", ([StatisticsRelevanceTable()]))
@pytest.mark.parametrize(
    "model",
    [
        DecisionTreeRegressor(random_state=42),
    ],
)
def test_pipeline_backtest_inverse_transform(sinusoid_ts, model, relevance_table):
    ts = sinusoid_ts

    pipeline = Pipeline(
        model=ProphetModel(),
        transforms=[
            LogTransform(in_column="feature_1"),
            FilterFeaturesTransform(exclude=["feature_1"], return_features=True),
        ],
        horizon=10,
    )
    metrics_df, forecast_df, fold_info_df = pipeline.backtest(ts=ts, metrics=[MAE()], aggregate_metrics=True)
    assert metrics_df["MAE"].iloc[0] < 2
