import pytest
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import TheilSenRegressor

from etna.analysis import plot_trend
from etna.transforms import ChangePointsTrendTransform
from etna.transforms import LinearTrendTransform
from etna.transforms import STLTransform
from etna.transforms import TheilSenTrendTransform


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
    plot_trend(ts=example_tsdf, trend_transform=ChangePointsTrendTransform(in_column="target"))


@pytest.mark.parametrize("period", (7, 30))
def test_plot_stl(example_tsdf, period):
    plot_trend(ts=example_tsdf, trend_transform=STLTransform(in_column="target", period=period))
