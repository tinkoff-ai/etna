import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import TheilSenRegressor

from etna.analysis import get_residuals
from etna.analysis import plot_residuals
from etna.analysis import plot_trend
from etna.analysis.plotters import _get_labels_names
from etna.analysis.plotters import _validate_intersecting_segments
from etna.datasets import TSDataset
from etna.metrics import MAE
from etna.models import LinearPerSegmentModel
from etna.pipeline import Pipeline
from etna.transforms import BinsegTrendTransform
from etna.transforms import LagTransform
from etna.transforms import LinearTrendTransform
from etna.transforms import STLTransform
from etna.transforms import TheilSenTrendTransform


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


@pytest.mark.parametrize(
    "poly_degree, trend_transform_class",
    (
        [1, LinearTrendTransform],
        [2, LinearTrendTransform],
        [1, TheilSenTrendTransform],
        [2, TheilSenTrendTransform],
    ),
)
def test_plot_trend(poly_degree, example_tsdf, trend_transform_class):
    plot_trend(ts=example_tsdf, trend_transform=trend_transform_class(in_column="target", poly_degree=poly_degree))


@pytest.mark.parametrize("detrend_model", (TheilSenRegressor(), LinearRegression()))
def test_plot_bin_seg(example_tsdf, detrend_model):
    plot_trend(ts=example_tsdf, trend_transform=BinsegTrendTransform(in_column="target", detrend_model=detrend_model))


@pytest.mark.parametrize("period", (7, 30))
def test_plot_stl(example_tsdf, period):
    plot_trend(ts=example_tsdf, trend_transform=STLTransform(in_column="target", period=period))


@pytest.mark.parametrize(
    "poly_degree, expect_values, trend_class",
    (
        [1, True, LinearTrendTransform],
        [2, False, LinearTrendTransform],
        [1, True, TheilSenTrendTransform],
        [2, False, TheilSenTrendTransform],
    ),
)
def test_get_labels_names_linear_coeffs(example_tsdf, poly_degree, expect_values, trend_class):
    ln_tr = trend_class(in_column="target", poly_degree=poly_degree)
    example_tsdf.fit_transform([ln_tr])
    segments = example_tsdf.segments
    _, linear_coeffs = _get_labels_names([ln_tr], segments)
    if expect_values:
        assert list(linear_coeffs.values()) != ["", ""]
    else:
        assert list(linear_coeffs.values()) == ["", ""]


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
