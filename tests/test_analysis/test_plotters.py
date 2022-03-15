import numpy as np
import pandas as pd
import pytest

from etna.analysis import get_residuals
from etna.analysis import plot_residuals
from etna.datasets import TSDataset
from etna.metrics import MAE
from etna.models import LinearPerSegmentModel
from etna.pipeline import Pipeline
from etna.transforms import LagTransform


@pytest.fixture
def residuals():
    timestamp = pd.date_range("2020-01-01", periods=100, freq="D")
    df = pd.DataFrame(
        {
            "timestamp": timestamp.tolist() * 2,
            "segment": ["segment_0"] * len(timestamp) + ["segment_1"] * len(timestamp),
            "target": np.arange(len(timestamp)).tolist() + (np.arange(len(timestamp)) + 1).tolist(),
        }
    )
    df_wide = TSDataset.to_dataset(df)
    ts = TSDataset(df=df_wide, freq="D")

    forecast_df = ts[timestamp[10:], :, :]
    forecast_df.loc[:, pd.IndexSlice["segment_0", "target"]] = -1
    forecast_df.loc[:, pd.IndexSlice["segment_1", "target"]] = 1

    residuals_df = ts[timestamp[10:], :, :]
    residuals_df.loc[:, pd.IndexSlice["segment_0", "target"]] += 1
    residuals_df.loc[:, pd.IndexSlice["segment_1", "target"]] -= 1

    return residuals_df, forecast_df, ts


def test_get_residuals(residuals):
    """Test that get_residuals finds residuals correctly."""
    residuals_df, forecast_df, ts = residuals
    actual_residuals = get_residuals(forecast_df=forecast_df, ts=ts)
    assert actual_residuals.to_pandas().equals(residuals_df)


def test_get_residuals_not_matching_lengths(residuals):
    """Test that get_residuals fails to find residuals correctly if ts hasn't answers."""
    residuals_df, forecast_df, ts = residuals
    ts = TSDataset(df=ts[ts.index[:-10], :, :], freq="D")
    with pytest.raises(KeyError):
        _ = get_residuals(forecast_df=forecast_df, ts=ts)


def test_get_residuals_not_matching_segments(residuals):
    """Test that get_residuals fails to find residuals correctly if segments of dataset and forecast differ."""
    residuals_df, forecast_df, ts = residuals
    columns_frame = forecast_df.columns.to_frame()
    columns_frame["segment"] = ["segment_0", "segment_3"]
    forecast_df.columns = pd.MultiIndex.from_frame(columns_frame)
    with pytest.raises(KeyError, match="Segments of `ts` and `forecast_df` should be the same"):
        _ = get_residuals(forecast_df=forecast_df, ts=ts)


def test_plot_residuals_fails_unkown_feature(example_tsdf):
    """Test that plot_residuals fails if meet unknown feature."""
    pipeline = Pipeline(
        model=LinearPerSegmentModel(), transforms=[LagTransform(in_column="target", lags=[5, 6, 7])], horizon=5
    )
    metrics, forecast_df, info = pipeline.backtest(ts=example_tsdf, metrics=[MAE()], n_folds=3)
    with pytest.raises(ValueError, match="Given feature isn't present in the dataset"):
        plot_residuals(forecast_df=forecast_df, ts=example_tsdf, feature="unkown_feature")
