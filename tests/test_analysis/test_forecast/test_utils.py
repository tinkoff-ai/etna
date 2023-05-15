import numpy as np
import pandas as pd
import pytest

from etna.analysis import get_residuals
from etna.analysis.forecast.utils import _validate_intersecting_segments
from etna.datasets import TSDataset


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


@pytest.fixture
def residuals_with_components(residuals):
    residuals_df, forecast_df, ts = residuals
    df_wide = ts.to_pandas()
    df_component_1 = df_wide.rename(columns={"target": "component_1"}, level="feature")
    df_component_2 = df_wide.rename(columns={"target": "component_2"}, level="feature")
    df_component_1.loc[:, pd.IndexSlice[:, "component_1"]] *= 0.7
    df_component_2.loc[:, pd.IndexSlice[:, "component_2"]] *= 0.3
    df_components = pd.concat([df_component_1, df_component_2], axis=1)
    ts.add_target_components(df_components)
    return residuals_df, forecast_df, ts


def test_get_residuals(residuals):
    """Test that get_residuals finds residuals correctly."""
    residuals_df, forecast_df, ts = residuals
    actual_residuals = get_residuals(forecast_df=forecast_df, ts=ts)
    assert actual_residuals.to_pandas().equals(residuals_df)


def test_get_residuals_with_components(residuals_with_components):
    """Test that get_residuals finds residuals correctly in case of target components presence."""
    residuals_df, forecast_df, ts = residuals_with_components
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


@pytest.mark.parametrize(
    "fold_numbers",
    [
        pd.Series([0, 0, 1, 1, 2, 2], index=pd.date_range("2020-01-01", periods=6, freq="D")),
        pd.Series([0, 0, 1, 1, 2, 2], index=pd.date_range("2020-01-01", periods=6, freq="2D")),
        pd.Series([2, 2, 0, 0, 1, 1], index=pd.date_range("2020-01-01", periods=6, freq="D")),
        pd.Series(
            [0, 0, 1, 1],
            index=[
                pd.Timestamp("2020-01-01"),
                pd.Timestamp("2020-01-02"),
                pd.Timestamp("2020-01-05"),
                pd.Timestamp("2020-01-06"),
            ],
        ),
    ],
)
def test_validate_intersecting_segments_ok(fold_numbers):
    _validate_intersecting_segments(fold_numbers)


@pytest.mark.parametrize(
    "fold_numbers",
    [
        pd.Series(
            [0, 0, 1, 1],
            index=[
                pd.Timestamp("2020-01-01"),
                pd.Timestamp("2020-01-02"),
                pd.Timestamp("2020-01-01"),
                pd.Timestamp("2020-01-02"),
            ],
        ),
        pd.Series([0, 0, 1, 1, 0, 0], index=pd.date_range("2020-01-01", periods=6, freq="D")),
        pd.Series(
            [0, 0, 1, 1],
            index=[
                pd.Timestamp("2020-01-01"),
                pd.Timestamp("2020-01-02"),
                pd.Timestamp("2020-01-02"),
                pd.Timestamp("2020-01-03"),
            ],
        ),
        pd.Series(
            [1, 1, 0, 0],
            index=[
                pd.Timestamp("2020-01-01"),
                pd.Timestamp("2020-01-02"),
                pd.Timestamp("2020-01-02"),
                pd.Timestamp("2020-01-03"),
            ],
        ),
        pd.Series(
            [0, 0, 1, 1],
            index=[
                pd.Timestamp("2020-01-01"),
                pd.Timestamp("2020-01-05"),
                pd.Timestamp("2020-01-03"),
                pd.Timestamp("2020-01-08"),
            ],
        ),
        pd.Series(
            [1, 1, 0, 0],
            index=[
                pd.Timestamp("2020-01-01"),
                pd.Timestamp("2020-01-05"),
                pd.Timestamp("2020-01-03"),
                pd.Timestamp("2020-01-08"),
            ],
        ),
    ],
)
def test_validate_intersecting_segments_fail(fold_numbers):
    with pytest.raises(ValueError):
        _validate_intersecting_segments(fold_numbers)
