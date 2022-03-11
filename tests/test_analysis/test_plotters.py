import numpy as np
import pandas as pd
import pytest

from etna.analysis import get_residuals
from etna.analysis import plot_residuals
from etna.metrics import MAE
from etna.models import LinearPerSegmentModel
from etna.pipeline import Pipeline
from etna.transforms import LagTransform


@pytest.mark.parametrize("ts_fixture", ["example_tsdf", "example_reg_tsds"])
def test_get_residuals(ts_fixture, request):
    """Test that get_residuals finds residuals correctly."""
    ts = request.getfixturevalue(ts_fixture)
    pipeline = Pipeline(
        model=LinearPerSegmentModel(), transforms=[LagTransform(in_column="target", lags=[5, 6, 7])], horizon=5
    )
    metrics, forecast_df, info = pipeline.backtest(ts=ts, metrics=[MAE()], n_folds=3)
    residuals = get_residuals(forecast_df=forecast_df, ts=ts)

    assert np.all(residuals.columns == ts.columns)
    assert set(residuals.to_pandas().index) == set(forecast_df.index)
    df = ts.to_pandas()
    for segment in ts.segments:
        true_segment = df.loc[forecast_df.index, pd.IndexSlice[segment, :]][segment]["target"]
        forecast_segment = forecast_df.loc[:, pd.IndexSlice[segment, :]][segment]["target"]
        residuals_segment = residuals[:, segment, "target"]
        assert np.all(residuals_segment == (true_segment - forecast_segment))


def test_plot_residuals_fails_unkown_feature(example_tsdf):
    """Test that plot_residuals fails if meet unknown feature."""
    pipeline = Pipeline(
        model=LinearPerSegmentModel(), transforms=[LagTransform(in_column="target", lags=[5, 6, 7])], horizon=5
    )
    metrics, forecast_df, info = pipeline.backtest(ts=example_tsdf, metrics=[MAE()], n_folds=3)
    with pytest.raises(ValueError, match="Given feature isn't present in the dataset"):
        plot_residuals(forecast_df=forecast_df, ts=example_tsdf, feature="unkown_feature")
